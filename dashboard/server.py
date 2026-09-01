import http.server
import socketserver
import json
import os
import subprocess
import urllib.parse

PORT = 8000
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DASHBOARD_DIR = os.path.join(PROJECT_DIR, 'dashboard')
VENV_PYTHON = os.path.join(PROJECT_DIR, '.venv', 'bin', 'python')

class DashboardRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Always serve from the dashboard folder for static files
        super().__init__(*args, directory=DASHBOARD_DIR, **kwargs)

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path == "/api/environments":
            self.handle_environments()
        elif path == "/api/results":
            self.handle_results()
        elif path.startswith("/api/image/"):
            self.handle_serve_image(path)
        else:
            super().do_GET()

    def do_POST(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path == "/api/run_experiment":
            self.handle_run_experiment()
        else:
            self.send_error(404, "Not Found")

    def handle_environments(self):
        try:
            # Let's run a quick python script using venv to get registered environments
            script = "from statisticalrl_environments.register import registerStatisticalRLenvironments; print(list(registerStatisticalRLenvironments.keys()))"
            env = os.environ.copy()
            env['PYTHONPATH'] = os.path.join(PROJECT_DIR, 'environments', 'src')
            result = subprocess.run(
                [VENV_PYTHON, "-c", script],
                capture_output=True,
                text=True,
                cwd=PROJECT_DIR,
                env=env
            )
            if result.returncode == 0:
                envs = eval(result.stdout.strip())
                self.send_json({"status": "success", "environments": envs})
            else:
                self.send_json({"status": "error", "message": result.stderr})
        except Exception as e:
            self.send_json({"status": "error", "message": str(e)})

    def handle_results(self):
        try:
            results_mab_dir = os.path.join(PROJECT_DIR, 'results_mab')
            screenshots_dir = os.path.join(PROJECT_DIR, 'screenshots')
            
            results_mab = []
            if os.path.exists(results_mab_dir):
                results_mab = sorted([f for f in os.listdir(results_mab_dir) if f.endswith(('.png', '.pdf', '.txt'))])

            screenshots = []
            if os.path.exists(screenshots_dir):
                screenshots = sorted([f for f in os.listdir(screenshots_dir) if f.endswith('.png')])

            self.send_json({
                "status": "success",
                "results_mab": results_mab,
                "screenshots": screenshots
            })
        except Exception as e:
            self.send_json({"status": "error", "message": str(e)})

    def handle_serve_image(self, path):
        # Format: /api/image/results_mab/filename.png or /api/image/screenshots/filename.png
        parts = path.strip("/").split("/")
        if len(parts) >= 4 and parts[2] in ['results_mab', 'screenshots']:
            folder = parts[2]
            filename = "/".join(parts[3:])
            filepath = os.path.abspath(os.path.join(PROJECT_DIR, folder, filename))
            
            # Security check: ensure path is within PROJECT_DIR
            if filepath.startswith(PROJECT_DIR) and os.path.exists(filepath):
                # Serve file
                if filename.endswith('.png'):
                    self.send_response(200)
                    self.send_header('Content-type', 'image/png')
                    self.end_headers()
                    with open(filepath, 'rb') as f:
                        self.wfile.write(f.read())
                    return
                elif filename.endswith('.pdf'):
                    self.send_response(200)
                    self.send_header('Content-type', 'application/pdf')
                    self.end_headers()
                    with open(filepath, 'rb') as f:
                        self.wfile.write(f.read())
                    return
                elif filename.endswith('.txt'):
                    self.send_response(200)
                    self.send_header('Content-type', 'text/plain')
                    self.end_headers()
                    with open(filepath, 'rb') as f:
                        self.wfile.write(f.read())
                    return
            
        self.send_error(404, "File Not Found")

    def handle_run_experiment(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode('utf-8'))
        
        script_name = data.get('script') # e.g. "banditsxp_eps.py" or "experiments/src/lenient_ts_experiment.py"
        if script_name not in ['banditsxp_eps.py', 'experiments/src/lenient_ts_experiment.py', 'epsilon_animation.py', 'epsilon.py']:
            self.send_json({"status": "error", "message": "Unauthorized script name"})
            return

        script_path = os.path.join(PROJECT_DIR, script_name)
        if not os.path.exists(script_path):
            self.send_json({"status": "error", "message": f"Script {script_name} not found"})
            return

        try:
            # Run the script in the background or synchronously if quick. Since it can take some time, let's run it and capture output
            result = subprocess.run(
                [VENV_PYTHON, script_path],
                capture_output=True,
                text=True,
                cwd=PROJECT_DIR
            )
            self.send_json({
                "status": "success",
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            })
        except Exception as e:
            self.send_json({"status": "error", "message": str(e)})

    def send_json(self, data):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

if __name__ == "__main__":
    handler = DashboardRequestHandler
    # Allow address reuse to facilitate quick restarts
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"Serving dashboard at http://localhost:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
