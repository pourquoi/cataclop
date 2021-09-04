#!/bin/bash
cd /app
export APP_VENV=$(poetry env info --path) 
supervisord -n -c /etc/supervisor/conf.d/supervisord.conf