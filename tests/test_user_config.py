import unittest
import os
import shutil
import tempfile
from core import user_config

class TestUserConfig(unittest.TestCase):
    def setUp(self):
        # Use a temporary directory for config
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'user_settings.json')
        self.backup_path = os.path.join(self.temp_dir, 'user_settings_backup.json')
        user_config.CONFIG_DIR = self.temp_dir
        user_config.CONFIG_FILE_PATH = self.config_path
        user_config.BACKUP_FILE_PATH = self.backup_path

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_load_default_config_when_missing(self):
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
        config = user_config.load_config()
        self.assertEqual(config, user_config.DEFAULT_CONFIG)

    def test_save_and_load_config(self):
        data = user_config.DEFAULT_CONFIG.copy()
        data['first_run_complete'] = True
        user_config.save_config(data)
        loaded = user_config.load_config()
        self.assertEqual(loaded['first_run_complete'], True)

    def test_backup_created_on_save(self):
        # Save initial config
        user_config.save_config(user_config.DEFAULT_CONFIG)
        # Save again to trigger backup
        user_config.save_config(user_config.DEFAULT_CONFIG)
        self.assertTrue(os.path.exists(self.backup_path))

    def test_validate_config_fills_missing_keys(self):
        partial = {'first_run_complete': True}
        validated = user_config.validate_config(partial)
        for key in user_config.DEFAULT_CONFIG:
            self.assertIn(key, validated)

    def test_load_config_with_invalid_json(self):
        with open(self.config_path, 'w') as f:
            f.write('{ invalid json }')
        config = user_config.load_config()
        self.assertEqual(config, user_config.DEFAULT_CONFIG)

if __name__ == '__main__':
    unittest.main()
