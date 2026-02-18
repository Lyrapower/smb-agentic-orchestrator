smb-agentic-orchestrator/
  README.md
  requirements.txt
  .env.example
  .gitignore

  app/
    __init__.py
    main.py                 # FastAPI入口
    settings.py             # 配置读取（env）
    models.py               # Pydantic请求/响应模型
    router.py               # 意图路由（最小规则版）
    audit.py                # 审计日志写入/读取
    tools/
      __init__.py
      scheduler.py          # “预约工具”（先mock）
      notifications.py      # “短信工具”（先mock）
    static/
      app.js                # 前端逻辑
      styles.css            # 简单样式
    templates/
      index.html            # 单页UI
    data/
      audit.jsonl           # 运行时生成（别手建）

  screenshots/
    ui.png
    audit.png
    (放你之后截图)
