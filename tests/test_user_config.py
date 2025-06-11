import unittest
import os
import shutil
import tempfile
from core import user_config

class TestUserConfig(unittest.TestCase):
    def setUp(self):
        # Use a temporary directory for config
        """
        Sets up a temporary directory and config file paths for isolated test execution.
        
        Overrides the user_config module's configuration directory and file path variables to use temporary locations, ensuring tests do not affect real user data.
        """
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'user_settings.json')
        self.backup_path = os.path.join(self.temp_dir, 'user_settings_backup.json')
        user_config.CONFIG_DIR = self.temp_dir
        user_config.CONFIG_FILE_PATH = self.config_path
        user_config.BACKUP_FILE_PATH = self.backup_path

    def tearDown(self):
        """
        Removes the temporary directory and its contents after each test.
        """
        shutil.rmtree(self.temp_dir)

    def test_load_default_config_when_missing(self):
        """
        Tests that the default configuration is loaded when the config file does not exist.
        """
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
        config = user_config.load_config()
        self.assertEqual(config, user_config.DEFAULT_CONFIG)

    def test_save_and_load_config(self):
        """
        Tests that saving a modified configuration and then loading it returns the updated values.
        
        Ensures that changes to the configuration, such as setting 'first_run_complete' to True, are correctly persisted and retrieved.
        """
        data = user_config.DEFAULT_CONFIG.copy()
        data['first_run_complete'] = True
        user_config.save_config(data)
        loaded = user_config.load_config()
        self.assertEqual(loaded['first_run_complete'], True)

    def test_backup_created_on_save(self):
        # Save initial config
        """
        Tests that a backup file is created when saving the configuration multiple times.
        
        Saves the configuration twice and asserts that a backup file exists at the expected backup path after the second save.
        """
        user_config.save_config(user_config.DEFAULT_CONFIG)
        # Save again to trigger backup
        user_config.save_config(user_config.DEFAULT_CONFIG)
        self.assertTrue(os.path.exists(self.backup_path))

    def test_validate_config_fills_missing_keys(self):
        """
        Tests that validate_config fills in missing keys from the default configuration.
        
        Ensures that when a partial configuration dictionary is provided, validate_config
        returns a dictionary containing all keys from the default configuration.
        """
        partial = {'first_run_complete': True}
        validated = user_config.validate_config(partial)
        for key in user_config.DEFAULT_CONFIG:
            self.assertIn(key, validated)

    def test_load_config_with_invalid_json(self):
        """
        Tests that loading the configuration with invalid JSON content returns the default configuration.
        """
        with open(self.config_path, 'w') as f:
            f.write('{ invalid json }')
        config = user_config.load_config()
        self.assertEqual(config, user_config.DEFAULT_CONFIG)

if __name__ == '__main__':
    unittest.main()
