#!/usr/bin/env python3
"""
Simple HTTP server with proper MIME types for ES6 modules
"""
import http.server
import socketserver
import os

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_my_headers()
        http.server.SimpleHTTPRequestHandler.end_headers(self)

    def send_my_headers(self):
        if self.path.endswith('.js'):
            self.send_header("Content-Type", "application/javascript")
        elif self.path.endswith('.mjs'):
            self.send_header("Content-Type", "application/javascript")

if __name__ == '__main__':
    PORT = 3000
    with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        print(f"Server running at http://localhost:{PORT}/")
        httpd.serve_forever()