import subprocess


def test_cli_help():
    # Run the CLI --help to ensure it loads without runtime errors
    cmd = ["python", "tradebot.py", "--help"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "TradeBot v2.0" in res.stdout
