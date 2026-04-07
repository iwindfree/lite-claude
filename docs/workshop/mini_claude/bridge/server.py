"""
브리지 서버 — IDE 통합을 위한 WebSocket 기반 메시징 시스템

Claude Code는 VS Code, JetBrains 등의 IDE와 WebSocket을 통해 통신합니다.
이 브리지 시스템은 다음을 처리합니다:
- 사용자 메시지 수신 (IDE → 에이전트)
- 어시스턴트 응답 전송 (에이전트 → IDE)
- 제어 요청/응답 (권한 확인 등)
- 메시지 중복 제거

참조:
- src/bridge/replBridge.ts        — WebSocket 서버 관리
- src/bridge/bridgeMessaging.ts   — 메시지 타입 및 프로토콜
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, Literal


# ─── 메시지 타입 ─────────────────────────────────────────────────

@dataclass
class BridgeMessage:
    """
    브리지 메시지 — Claude Code의 BridgeMessage에 대응

    type:
        - 'user': IDE에서 에이전트로 보내는 사용자 입력
        - 'assistant': 에이전트에서 IDE로 보내는 응답
        - 'control_request': 에이전트가 IDE에 권한 확인 요청
        - 'control_response': IDE가 에이전트에 권한 결정 응답
    content: 메시지 내용
    request_id: 요청-응답 매칭용 ID (control_request/response에 사용)
    metadata: 추가 메타데이터 (도구 이름, 상태 등)
    """
    type: Literal["user", "assistant", "control_request", "control_response"]
    content: str
    request_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """JSON 문자열로 직렬화"""
        return json.dumps({
            "type": self.type,
            "content": self.content,
            "request_id": self.request_id,
            "metadata": self.metadata,
        }, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> BridgeMessage:
        """JSON 문자열에서 역직렬화"""
        data = json.loads(text)
        return cls(
            type=data.get("type", "user"),
            content=data.get("content", ""),
            request_id=data.get("request_id", ""),
            metadata=data.get("metadata", {}),
        )


# ─── 메시지 핸들러 타입 ─────────────────────────────────────────

MessageHandler = Callable[[BridgeMessage], Awaitable[None]]


# ─── 브리지 서버 ────────────────────────────────────────────────

class BridgeServer:
    """
    WebSocket 기반 브리지 서버 — Claude Code의 replBridge에 대응

    IDE(클라이언트)와 에이전트(서버) 간의 양방향 통신을 관리합니다.
    단일 연결만 허용하는 간소화된 버전입니다.

    사용법:
        server = BridgeServer(host="localhost", port=8765)
        server.on_user_message = my_handler
        await server.start()
    """

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port

        # 메시지 핸들러
        self.on_user_message: MessageHandler | None = None
        self.on_control_response: MessageHandler | None = None

        # 내부 상태
        self._websocket = None
        self._server = None
        self._sent_ids: set[str] = set()  # 중복 전송 방지용
        self._pending_controls: dict[str, asyncio.Future] = {}  # 대기 중인 제어 요청

    async def start(self) -> None:
        """WebSocket 서버를 시작합니다."""
        try:
            import websockets
        except ImportError:
            raise ImportError(
                "websockets 라이브러리가 필요합니다: pip install websockets"
            )

        self._server = await websockets.serve(
            self._handle_connection,
            self.host,
            self.port,
        )
        print(f"[Bridge] 서버 시작: ws://{self.host}:{self.port}")

    async def stop(self) -> None:
        """서버를 정지합니다."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        self._websocket = None
        print("[Bridge] 서버 정지")

    async def _handle_connection(self, websocket, path=None) -> None:
        """
        새 WebSocket 연결을 처리합니다.

        Claude Code처럼 단일 연결만 유지합니다.
        새 연결이 오면 이전 연결을 대체합니다.
        """
        self._websocket = websocket
        print("[Bridge] 클라이언트 연결됨")

        try:
            async for raw_message in websocket:
                await self._handle_raw_message(raw_message)
        except Exception as e:
            print(f"[Bridge] 연결 종료: {e}")
        finally:
            if self._websocket is websocket:
                self._websocket = None
            print("[Bridge] 클라이언트 연결 해제")

    async def _handle_raw_message(self, raw: str) -> None:
        """수신한 원시 메시지를 파싱하고 적절한 핸들러에 전달합니다."""
        try:
            msg = BridgeMessage.from_json(raw)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[Bridge] 메시지 파싱 실패: {e}")
            return

        if msg.type == "user":
            if self.on_user_message:
                await self.on_user_message(msg)

        elif msg.type == "control_response":
            # 대기 중인 제어 요청에 응답
            if msg.request_id in self._pending_controls:
                self._pending_controls[msg.request_id].set_result(msg)
            if self.on_control_response:
                await self.on_control_response(msg)

    async def send_assistant_message(self, content: str, metadata: dict | None = None) -> None:
        """어시스턴트 응답을 IDE에 전송합니다."""
        msg = BridgeMessage(
            type="assistant",
            content=content,
            request_id=str(uuid.uuid4()),
            metadata=metadata or {},
        )
        await self._send(msg)

    async def request_control(
        self,
        content: str,
        metadata: dict | None = None,
        timeout: float = 30.0,
    ) -> BridgeMessage | None:
        """
        IDE에 제어 요청을 보내고 응답을 대기합니다.

        예: 도구 실행 권한 확인
            response = await server.request_control(
                "Bash 도구로 'rm -rf /' 실행을 허용하시겠습니까?",
                metadata={"tool": "Bash", "command": "rm -rf /"},
            )
            if response and response.content == "allow":
                # 실행 허용
        """
        request_id = str(uuid.uuid4())
        msg = BridgeMessage(
            type="control_request",
            content=content,
            request_id=request_id,
            metadata=metadata or {},
        )

        # Future를 등록하고 응답 대기
        future: asyncio.Future[BridgeMessage] = asyncio.get_event_loop().create_future()
        self._pending_controls[request_id] = future

        try:
            await self._send(msg)
            response = await asyncio.wait_for(future, timeout=timeout)
            return response
        except asyncio.TimeoutError:
            print(f"[Bridge] 제어 요청 타임아웃: {request_id}")
            return None
        finally:
            self._pending_controls.pop(request_id, None)

    async def _send(self, msg: BridgeMessage) -> None:
        """메시지를 WebSocket으로 전송합니다. 중복 전송을 방지합니다."""
        if not self._websocket:
            print("[Bridge] 전송 실패: 연결된 클라이언트 없음")
            return

        # 중복 전송 방지 (echo dedup)
        msg_id = msg.request_id
        if msg_id and msg_id in self._sent_ids:
            return
        if msg_id:
            self._sent_ids.add(msg_id)
            # 메모리 관리: 오래된 ID 제거 (최대 1000개 유지)
            if len(self._sent_ids) > 1000:
                # set은 순서가 없으므로 전체 클리어 후 현재 ID만 유지
                self._sent_ids = {msg_id}

        try:
            await self._websocket.send(msg.to_json())
        except Exception as e:
            print(f"[Bridge] 전송 실패: {e}")

    @property
    def is_connected(self) -> bool:
        """클라이언트가 연결되어 있는지 확인"""
        return self._websocket is not None


# ─── 편의 함수: 간단한 에코 브리지 ──────────────────────────────

async def run_echo_bridge(host: str = "localhost", port: int = 8765) -> None:
    """
    디버깅용 에코 브리지. 수신한 메시지를 그대로 돌려보냅니다.

    사용법:
        python -m mini_claude.bridge.server
    """
    server = BridgeServer(host=host, port=port)

    async def echo_handler(msg: BridgeMessage) -> None:
        print(f"[Echo] 수신: {msg.content[:100]}")
        await server.send_assistant_message(
            content=f"[에코] {msg.content}",
            metadata={"echo": True},
        )

    server.on_user_message = echo_handler
    await server.start()

    # 서버가 종료될 때까지 대기
    try:
        await asyncio.Future()  # 무한 대기
    except asyncio.CancelledError:
        await server.stop()


if __name__ == "__main__":
    asyncio.run(run_echo_bridge())
