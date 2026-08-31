---
name: Streamlit preview WebSockets
description: Replit proxy behavior required for Streamlit’s browser WebSocket session in the ReturnShield preview.
---

Replit’s proxied Streamlit preview can reject the forwarded host and origin even when the HTTP page and local server are healthy. The preview workflow must launch Streamlit with both `server.enableCORS` and `server.enableXsrfProtection` disabled.

**Why:** Streamlit’s default origin protection sees the proxy host/origin as disallowed, producing repeated client-side `WebSocket onerror` messages while ordinary HTTP requests continue to work.

**How to apply:** Use these settings only on the development preview workflow; keep production deployments behind their intended authentication and proxy protections.