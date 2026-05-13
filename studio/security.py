from __future__ import annotations

import secrets

from flask import flash, jsonify, redirect, request, session


def init_csrf(app) -> None:
    @app.context_processor
    def inject_csrf_token():
        return {"csrf_token": get_csrf_token}

    @app.before_request
    def protect_against_csrf():
        if request.method not in {"POST", "PUT", "PATCH", "DELETE"}:
            return None

        sent_token = request.headers.get("X-CSRF-Token") or request.form.get("csrf_token")
