from __future__ import annotations

import socket
import struct
from dataclasses import dataclass
from typing import BinaryIO

from config import DASHBOARD_PORT, ROBOT_HOST, SOCKET_TIMEOUT_S, STATE_PORT

ROBOT_STATE_MESSAGE_TYPE = 16


@dataclass(frozen=True)
class RobotStateSubPacket:
    # 30001 报文里的一个子报文。由于手册正文只给了顶层结构，这里先保留原始负载。
    packet_length: int
    packet_type: int
    payload: bytes


@dataclass(frozen=True)
class RobotStatePacket:
    # 30001 顶层机器人状态报文。
    packet_length: int
    packet_type: int
    sub_packets: list[RobotStateSubPacket]


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("socket closed while receiving data")
        chunks.extend(chunk)
    return bytes(chunks)


class DashboardClient:
    """
    CS612A Dashboard 驱动。

    手册依据：
    1. 29999 端口用于 Dashboard 命令控制。
    2. 命令必须以 '\\n' 作为结束标识。
    3. 连接成功后可以发送 help / usage 等文本命令。
    """

    def __init__(self, host: str, port: int = DASHBOARD_PORT, timeout: float = SOCKET_TIMEOUT_S) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock: socket.socket | None = None

    def connect(self) -> str:
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self.sock.settimeout(self.timeout)
        return self._read_available()

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def send_command(self, command: str) -> str:
        if self.sock is None:
            raise RuntimeError("dashboard socket is not connected")
        wire = command.rstrip("\n") + "\n"
        self.sock.sendall(wire.encode("utf-8"))
        return self._read_available()

    def help(self) -> str:
        return self.send_command("help")

    def usage(self) -> str:
        return self.send_command("usage")

    def _read_available(self) -> str:
        if self.sock is None:
            return ""

        chunks: list[bytes] = []
        while True:
            try:
                chunk = self.sock.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
                if len(chunk) < 4096:
                    break
            except socket.timeout:
                break
        return b"".join(chunks).decode("utf-8", errors="replace")

    def __enter__(self) -> "DashboardClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class RobotStateClient:
    """
    CS612A 30001 状态流驱动。

    手册依据：
    1. 30001 为主端口，约 10Hz 输出机器人状态数据。
    2. 顶层报文结构是：4 字节长度 + 1 字节类型 + 若干子报文。
    3. 报文类型 16 表示机器人状态报文。
    """

    def __init__(self, host: str, port: int = STATE_PORT, timeout: float = SOCKET_TIMEOUT_S) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock: socket.socket | None = None

    def connect(self) -> None:
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self.sock.settimeout(self.timeout)

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def read_packet(self) -> RobotStatePacket:
        if self.sock is None:
            raise RuntimeError("state socket is not connected")

        header = _recv_exact(self.sock, 5)
        packet_length = struct.unpack(">I", header[:4])[0]
        packet_type = header[4]
        body = _recv_exact(self.sock, packet_length - 5)

        if packet_type != ROBOT_STATE_MESSAGE_TYPE:
            raise ValueError(f"unexpected packet type: {packet_type}")

        sub_packets = self._parse_sub_packets(body)
        return RobotStatePacket(
            packet_length=packet_length,
            packet_type=packet_type,
            sub_packets=sub_packets,
        )

    def _parse_sub_packets(self, body: bytes) -> list[RobotStateSubPacket]:
        sub_packets: list[RobotStateSubPacket] = []
        offset = 0
        while offset + 5 <= len(body):
            sub_length = struct.unpack(">I", body[offset : offset + 4])[0]
            sub_type = body[offset + 4]
            payload_start = offset + 5
            payload_end = offset + sub_length
            if sub_length < 5 or payload_end > len(body):
                break
            sub_packets.append(
                RobotStateSubPacket(
                    packet_length=sub_length,
                    packet_type=sub_type,
                    payload=body[payload_start:payload_end],
                )
            )
            offset = payload_end
        return sub_packets

    def __enter__(self) -> "RobotStateClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class ScriptClient:
    """
    CS612A 30001 脚本发送驱动。

    手册依据：
    1. 30001 端口会接收并执行正确的脚本。
    2. 脚本需以 `def name():\\n` 或 `sec name():\\n` 开头，以 `\\nend` 结尾。
    3. `sec` 脚本可与正在执行的 `def` 脚本并行运行，但不允许运动指令。
    """

    def __init__(self, host: str, port: int = STATE_PORT, timeout: float = SOCKET_TIMEOUT_S) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout

    def send_script(self, script: str) -> None:
        normalized = self._normalize_script(script)
        with socket.create_connection((self.host, self.port), timeout=self.timeout) as sock:
            sock.settimeout(self.timeout)
            sock.sendall(normalized.encode("utf-8"))

    def send_def(self, name: str, body: str) -> None:
        script = self.build_script(name=name, body=body, kind="def")
        self.send_script(script)

    def send_sec(self, name: str, body: str) -> None:
        script = self.build_script(name=name, body=body, kind="sec")
        self.send_script(script)

    @staticmethod
    def build_script(name: str, body: str, kind: str = "def") -> str:
        if kind not in {"def", "sec"}:
            raise ValueError("kind must be 'def' or 'sec'")

        lines = [f"{kind} {name}():"]
        for raw_line in body.splitlines():
            stripped = raw_line.rstrip()
            if stripped:
                lines.append(f"    {stripped}")
        lines.append("end")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _normalize_script(script: str) -> str:
        stripped = script.strip("\n")
        if not stripped.endswith("end"):
            raise ValueError("script must end with 'end'")
        return stripped + "\n"


class CS612ADriver:
    """
    一份轻量的 CS612A 驱动封装。

    提供三类能力：
    1. Dashboard 文本命令
    2. 30001 状态流接收
    3. 30001 脚本发送
    """

    def __init__(self, host: str, dashboard_port: int = DASHBOARD_PORT, state_port: int = STATE_PORT) -> None:
        self.host = host
        self.dashboard = DashboardClient(host, port=dashboard_port)
        self.state = RobotStateClient(host, port=state_port)
        self.script = ScriptClient(host, port=state_port)


def save_packet_summary(packet: RobotStatePacket, stream: BinaryIO) -> None:
    # 方便调试 30001 状态流，把顶层包和子包信息写到文件里。
    lines = [
        f"packet_length={packet.packet_length}",
        f"packet_type={packet.packet_type}",
        f"sub_packet_count={len(packet.sub_packets)}",
    ]
    for index, sub_packet in enumerate(packet.sub_packets):
        lines.append(
            f"sub[{index}] length={sub_packet.packet_length} type={sub_packet.packet_type} payload_bytes={len(sub_packet.payload)}"
        )
    stream.write(("\n".join(lines) + "\n").encode("utf-8"))


if __name__ == "__main__":
    # 把 host 改成机器人 IP 后可直接做最小连通性测试。
    driver = CS612ADriver(ROBOT_HOST)

    try:
        banner = driver.dashboard.connect()
        print("dashboard connected:")
        print(banner.strip())
        print(driver.dashboard.help())
    finally:
        driver.dashboard.close()
