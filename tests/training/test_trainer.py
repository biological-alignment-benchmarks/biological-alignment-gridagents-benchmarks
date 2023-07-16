import subprocess

from tests.test_config import constants


def test_training_pipeline():
    const = constants()
    ret = subprocess.run(["python", "-m", f"{const.PROJECT}"])
    assert ret.returncode == 0, "Trainer caused an error!"
