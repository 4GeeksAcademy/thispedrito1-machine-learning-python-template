"""
Comprehensive tests for src/utils.py - db_connect()
"""
import os
import sys
from unittest.mock import patch, MagicMock
import pytest

# Add the src directory to the Python path so we can import utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import utils


class TestDbConnectSuccess:
    """Tests for successful database connection scenarios."""

    def test_returns_engine_when_database_url_set(self):
        """Engine should be returned when DATABASE_URL is properly configured."""
        with patch.dict(os.environ, {'DATABASE_URL': 'sqlite:///:memory:'}), \
             patch('utils.create_engine') as mock_create_engine:
            mock_engine = MagicMock()
            mock_create_engine.return_value = mock_engine

            result = utils.db_connect()

            assert result is mock_engine
            mock_create_engine.assert_called_once_with('sqlite:///:memory:')
            mock_engine.connect.assert_called_once()

    def test_calls_connect_on_engine(self):
        """db_connect should call .connect() on the engine to verify connectivity."""
        with patch.dict(os.environ, {'DATABASE_URL': 'postgresql://user:pass@localhost:5432/db'}), \
             patch('utils.create_engine') as mock_create_engine:
            mock_engine = MagicMock()
            mock_create_engine.return_value = mock_engine

            utils.db_connect()

            mock_engine.connect.assert_called_once()

    def test_returns_same_engine_instance(self):
            """The function should return the engine created by create_engine."""
            with patch.dict(os.environ, {'DATABASE_URL': 'sqlite:///:memory:'}), \
                 patch('utils.create_engine') as mock_create_engine:
                mock_engine = MagicMock()
                mock_create_engine.return_value = mock_engine

                result = utils.db_connect()

                assert result == mock_engine


class TestDbConnectFailure:
    """Tests for failure scenarios."""

    def test_missing_database_url_returns_none(self):
        """When DATABASE_URL is missing, create_engine receives None and should raise."""
        env = {k: v for k, v in os.environ.items() if k != 'DATABASE_URL'}
        with patch.dict(os.environ, env, clear=True), \
             patch('utils.create_engine') as mock_create_engine:
            mock_create_engine.side_effect = TypeError("database URL is required")

            with pytest.raises(TypeError):
                utils.db_connect()

    def test_invalid_database_url_raises(self):
        """An invalid DATABASE_URL should propagate the underlying exception."""
        with patch.dict(os.environ, {'DATABASE_URL': 'not-a-valid-url'}), \
             patch('utils.create_engine') as mock_create_engine:
            mock_create_engine.side_effect = ValueError("invalid URL")

            with pytest.raises(ValueError):
                utils.db_connect()

    def test_connection_failure_raises(self):
        """If engine.connect() fails, the exception should propagate."""
        with patch.dict(os.environ, {'DATABASE_URL': 'sqlite:///:memory:'}), \
             patch('utils.create_engine') as mock_create_engine:
            mock_engine = MagicMock()
            mock_engine.connect.side_effect = ConnectionError("DB unreachable")
            mock_create_engine.return_value = mock_engine

            with pytest.raises(ConnectionError):
                utils.db_connect()

    def test_sqlalchemy_error_propagates(self):
        """A SQLAlchemy operational error should propagate to the caller."""
        from sqlalchemy.exc import OperationalError
        with patch.dict(os.environ, {'DATABASE_URL': 'sqlite:///:memory:'}), \
             patch('utils.create_engine') as mock_create_engine:
            mock_engine = MagicMock()
            mock_engine.connect.side_effect = OperationalError("stmt", {}, Exception("db error"))
            mock_create_engine.return_value = mock_engine

            with pytest.raises(OperationalError):
                utils.db_connect()


class TestDbConnectEnvironment:
    """Tests for environment variable handling."""

    def test_loads_dotenv_at_import_time(self):
        """utils module should call load_dotenv at import time."""
        import importlib
        with patch('dotenv.load_dotenv') as mock_load:
            if 'utils' in sys.modules:
                importlib.reload(sys.modules['utils'])
            else:
                importlib.import_module('utils')
            mock_load.assert_called()

    def test_passes_database_url_to_create_engine(self):
        """The DATABASE_URL from the environment should be passed to create_engine."""
        test_url = 'mysql+pymysql://root@localhost/test'
        with patch.dict(os.environ, {'DATABASE_URL': test_url}), \
             patch('utils.create_engine') as mock_create_engine:
            mock_engine = MagicMock()
            mock_create_engine.return_value = mock_engine

            utils.db_connect()

            called_args, _ = mock_create_engine.call_args
            assert called_args[0] == test_url


class TestDbConnectEdgeCases:
    """Tests for edge cases and unusual inputs."""

    def test_empty_string_database_url(self):
        """An empty DATABASE_URL should raise an error (treated as invalid)."""
        with patch.dict(os.environ, {'DATABASE_URL': ''}), \
             patch('utils.create_engine') as mock_create_engine:
            mock_create_engine.side_effect = TypeError("database URL is required")

            with pytest.raises(TypeError):
                utils.db_connect()

    def test_multiple_calls_create_multiple_engines(self):
        """Calling db_connect() multiple times should call create_engine each time."""
        with patch.dict(os.environ, {'DATABASE_URL': 'sqlite:///:memory:'}), \
             patch('utils.create_engine') as mock_create_engine:
            mock_engine = MagicMock()
            mock_create_engine.return_value = mock_engine

            utils.db_connect()
            utils.db_connect()
            utils.db_connect()

            assert mock_create_engine.call_count == 3
            assert mock_engine.connect.call_count == 3

    def test_special_characters_in_url(self):
        """URLs with special characters (e.g., passwords) should be passed through unchanged."""
        special_url = 'postgresql://user:p@ssw%3Aord@localhost:5432/db'
        with patch.dict(os.environ, {'DATABASE_URL': special_url}), \
             patch('utils.create_engine') as mock_create_engine:
            mock_engine = MagicMock()
            mock_create_engine.return_value = mock_engine

            utils.db_connect()

            mock_create_engine.assert_called_once_with(special_url)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
