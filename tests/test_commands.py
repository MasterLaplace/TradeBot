import unittest
from argparse import Namespace

from src.cli.commands import ListCommand


class TestListCommand(unittest.TestCase):
    def test_list_command_returns_zero(self):
        args = Namespace()
        cmd = ListCommand()
        rc = cmd.execute(args)
        self.assertEqual(rc, 0)


if __name__ == '__main__':
    unittest.main()
