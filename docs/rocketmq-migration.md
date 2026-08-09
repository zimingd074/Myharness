# Redis Streams → RocketMQ 迁移说明

## 改动范围

- 删除 Redis 容器、`redis` Python 依赖和 `EVOAGENT_REDIS_URL`。
- Docker Compose 新增 RocketMQ NameServer 与 Broker；应用通过
  `EVOAGENT_ROCKETMQ_NAMESERVER=rocketmq-namesrv:9876` 连接。
- 异步审查任务使用 `EvoAgentReview` Topic，Consumer Group 为
  `evoagent-workers`；同组 Worker 由 RocketMQ 负载均衡。
- 达到 `EVOAGENT_QUEUE_MAX_ATTEMPTS` 或发生不可恢复错误时，消息被确认并
  持久化发送至 `EvoAgentReviewDLQ`。任务库同步保存 DLQ 索引，因此原有
  `/api/queue/dead-letters` 与 `/v1/queue/dead-letters/replay` API 保持可用。

## 语义对照

| Redis Streams | RocketMQ 实现 |
| --- | --- |
| Consumer Group + `XACK` | PushConsumer 回调返回 `CONSUME_SUCCESS` |
| Pending Entry + `XAUTOCLAIM` 租约接管 | Broker 消费租约、超时和 Consumer Group 重平衡 |
| 应用重新 `XADD` 重试 | 回调返回 `RECONSUME_LATER`，由 broker 重投 |
| Redis Stream DLQ | RocketMQ `EvoAgentReviewDLQ` Topic + 任务库索引 |

消息仍按“至少一次”处理。Webhook delivery 的幂等由现有数据库 delivery-id
约束保证；任务处理及评论 upsert 必须继续以 task ID 作为幂等键。

## 验证步骤

```powershell
docker compose up --build
docker compose ps
```

应用启动后应显示队列后端为 `rocketmq`。执行测试套件会覆盖内存回退、DLQ
记录和 RocketMQ ACK/重试映射；集成测试需要本机 Docker 可用。
