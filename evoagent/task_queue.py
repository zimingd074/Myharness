"""Durable task delivery with RocketMQ ACK, retry, and dead-letter support."""
import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Optional


class PermanentTaskError(RuntimeError):
    """An error that must not be retried."""


class TaskQueue:
    TOPIC = "EvoAgentReview"
    DLQ_TOPIC = "EvoAgentReviewDLQ"
    GROUP = "evoagent-workers"

    def __init__(
        self, handler: Callable[[Dict], None], workers: int = 2,
        rocketmq_nameserver: str = "", max_attempts: int = 3,
        lease_seconds: int = 60,
        on_dead_letter: Optional[Callable[[Dict, str], None]] = None,
    ):
        self.handler = handler
        self.max_attempts = max_attempts
        # RocketMQ's broker owns the in-flight lease.  Keep this setting so the
        # existing configuration remains meaningful in the memory fallback.
        self.lease_seconds = lease_seconds
        self.on_dead_letter = on_dead_letter
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="evoagent-worker")
        self._memory_dlq = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._consumer = None
        self._producer = None
        if rocketmq_nameserver:
            self._start_rocketmq(rocketmq_nameserver, workers)

    def _start_rocketmq(self, nameserver: str, workers: int) -> None:
        try:
            from rocketmq.client import ConsumeStatus, Producer, PushConsumer
        except ImportError as exc:
            raise RuntimeError(
                "RocketMQ mode requires rocketmq-client-python and librocketmq"
            ) from exc
        self._consume_success = ConsumeStatus.CONSUME_SUCCESS
        self._reconsume_later = ConsumeStatus.RECONSUME_LATER
        self._producer = Producer("evoagent-dlq-producer")
        self._producer.set_name_server_address(nameserver)
        self._producer.start()
        self._consumer = PushConsumer(self.GROUP)
        self._consumer.set_name_server_address(nameserver)
        self._consumer.set_thread_count(workers)
        self._consumer.subscribe(self.TOPIC, self._rocketmq_callback)
        self._consumer.start()

    @property
    def backend(self) -> str:
        return "rocketmq" if self._consumer else "memory-acked"

    def submit(self, payload: Dict, message_id: str = "") -> str:
        envelope = {
            "message_id": message_id or str(payload.get("task_id") or uuid.uuid4()),
            "payload": payload,
            "submitted_at": time.time(),
        }
        if self._producer:
            self._send(self.TOPIC, envelope)
        else:
            self._executor.submit(self._deliver_memory, envelope, 1)
        return envelope["message_id"]

    def _send(self, topic: str, envelope: Dict) -> None:
        from rocketmq.client import Message

        message = Message(topic)
        message.set_keys(str(envelope["message_id"]))
        message.set_body(json.dumps(envelope, ensure_ascii=False).encode("utf-8"))
        self._producer.send_sync(message)

    def _rocketmq_callback(self, message):
        try:
            envelope = json.loads(message.body.decode("utf-8"))
            if not isinstance(envelope, dict) or "payload" not in envelope:
                raise ValueError("queue envelope must contain payload")
        except Exception as exc:
            self._dead_letter({"message_id": message.id, "payload": {}},
                              "invalid queue envelope: %s" % exc)
            return self._consume_success
        attempt = int(message.reconsume_times) + 1
        try:
            self.handler(envelope["payload"])
            return self._consume_success
        except PermanentTaskError as exc:
            self._dead_letter(envelope, str(exc), attempt)
            return self._consume_success
        except Exception as exc:
            if attempt >= self.max_attempts:
                self._dead_letter(envelope, str(exc), attempt)
                return self._consume_success
            return self._reconsume_later

    def _deliver_memory(self, envelope: Dict, attempt: int) -> None:
        try:
            self.handler(envelope["payload"])
        except PermanentTaskError as exc:
            self._dead_letter(envelope, str(exc), attempt)
        except Exception as exc:
            if attempt >= self.max_attempts:
                self._dead_letter(envelope, str(exc), attempt)
                return
            delay = min(2 ** (attempt - 1), 10)
            timer = threading.Timer(delay, self._submit_memory, args=(envelope, attempt + 1))
            timer.daemon = True
            timer.start()

    def _submit_memory(self, envelope: Dict, attempt: int) -> None:
        if not self._stop.is_set():
            self._executor.submit(self._deliver_memory, envelope, attempt)

    def _dead_letter(self, envelope: Dict, error: str, attempt: int = 1) -> None:
        item = {
            **envelope, "attempt": attempt, "error": error[:2000],
            "failed_at": time.time(),
        }
        if self._producer:
            self._send(self.DLQ_TOPIC, item)
        else:
            with self._lock:
                self._memory_dlq.append(item)
        if self.on_dead_letter:
            self.on_dead_letter(envelope.get("payload") or {}, item["error"])

    def dead_letters(self, limit: int = 100) -> list:
        with self._lock:
            return list(reversed(self._memory_dlq[-limit:]))

    def replay_dead_letter(self, message_id: str) -> bool:
        for item in self.dead_letters(500):
            if item.get("message_id") == message_id:
                self.submit(item.get("payload") or {}, message_id=message_id)
                return True
        return False

    def close(self, wait: bool = False) -> None:
        self._stop.set()
        if self._consumer:
            self._consumer.shutdown()
        if self._producer:
            self._producer.shutdown()
        self._executor.shutdown(wait=wait)
