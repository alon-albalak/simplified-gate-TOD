import json
import os


class simple_logger():
    def __init__(self, config):
        assert(config['log_path'] is not None)
        self.save_path = os.path.join(
            os.path.abspath(os.getcwd()), config['log_path'])

        if os.path.isfile(self.save_path):
            print(f"Loading log from {self.save_path}")
            self.logger = json.load(open(self.save_path))

        else:
            print(f"Initializing log at {self.save_path}")
            self.logger = {
                "training": [],
                "testing": {},
                "metadata": config,
                "errors": []
            }
            self.save()

    def save(self):
        with open(self.save_path, "w") as p:
            json.dump(self.logger, p, indent=2)
        print(f"Saved log at {self.save_path}")

    def training_update(self, status):
        self.logger['training'].append(status)
