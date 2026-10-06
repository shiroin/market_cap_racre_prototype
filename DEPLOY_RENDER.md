# Render deployment

This app is deployed as a Docker web service.

- Entrypoint: `start.sh`
- Streamlit host: `0.0.0.0`
- Port: Render-provided `$PORT` (fallback `10000`)
- Health check: `/_stcore/health`
- File watcher: disabled in production

A healthy startup log should include:

```text
Starting Streamlit on 0.0.0.0:<PORT>
You can now view your Streamlit app in your browser.
```

If Render still times out before the first line appears, verify that the service is using the repository Dockerfile and that no dashboard-level Docker Command override is configured.
