import json
from typing import Dict, Any, Optional


class DynamicAccessDict:
    """
    A helper class that provides dot notation access to dictionary items.
    This allows for accessing nested dictionary values using attribute syntax.
    """
    
    def __init__(self, data: Dict[str, Any]):
        self._data = data
    
    def __getattr__(self, name: str) -> Any:
        """
        Access dictionary items using dot notation.
        
        Args:
            name: The attribute/key name to access.
            
        Returns:
            The value, converted to DynamicAccessDict if it's a dictionary.
            
        Raises:
            AttributeError: If the key doesn't exist in the dictionary.
        """
        if name not in self._data:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
        value = self._data[name]
        
        # Convert nested dictionaries to DynamicAccessDict objects
        if isinstance(value, dict):
            return DynamicAccessDict(value)
        # Convert lists of dictionaries to lists of DynamicAccessDict objects
        elif isinstance(value, list):
            return [DynamicAccessDict(item) if isinstance(item, dict) else item for item in value]
        else:
            return value
    
    def __setattr__(self, name: str, value: Any) -> None:
        """
        Set dictionary items using dot notation.
        
        Args:
            name: The attribute/key name to set.
            value: The value to set.
        """
        if name == "_data":
            super().__setattr__(name, value)
        else:
            self._data[name] = value
    
    def __getitem__(self, key: str) -> Any:
        """
        Allow dictionary-style access.
        
        Args:
            key: The key to retrieve.
            
        Returns:
            The value associated with the key.
        """
        return self.__getattr__(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """
        Allow dictionary-style assignment.
        
        Args:
            key: The key to set.
            value: The value to associate with the key.
        """
        self.__setattr__(key, value)
    
    def __contains__(self, key: str) -> bool:
        """
        Allow 'in' operator.
        
        Args:
            key: The key to check.
            
        Returns:
            True if the key exists, False otherwise.
        """
        return key in self._data
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert back to a regular dictionary.
        
        Returns:
            A dictionary representation.
        """
        return self._data.copy()


class GameState:
    """
    A class that represents the state of a game, which can be loaded from a JSON file.
    This serves as an inverse of the object.to_dict() pattern.
    Properties can be accessed using dot notation (e.g., game.player.health).
    """
    
    def __init__(self, json_load_file_path: Optional[str] = None):
        """
        Initialize a GameState object, optionally loading from a JSON file.
        
        Args:
            json_load_file_path: Path to a JSON file to load the game state from.
                                If None, an empty game state is created.
        """
        self._data = {}
        
        if json_load_file_path:
            self.load_from_json(json_load_file_path)
    
    def load_from_json(self, file_path: str) -> None:
        """
        Load the game state from a JSON file.
        
        Args:
            file_path: Path to the JSON file to load.
            
        Raises:
            FileNotFoundError: If the specified file does not exist.
            json.JSONDecodeError: If the file contains invalid JSON.
        """
        try:
            with open(file_path, 'r') as file:
                self._data = json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find JSON file at {file_path}")
        except json.JSONDecodeError:
            raise json.JSONDecodeError(f"Invalid JSON format in file {file_path}", "", 0)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the game state.
        
        Args:
            key: The key to retrieve.
            default: The default value to return if the key is not found.
            
        Returns:
            The value associated with the key, or the default value if not found.
        """
        value = self._data.get(key, default)
        if isinstance(value, dict):
            return DynamicAccessDict(value)
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a value in the game state.
        
        Args:
            key: The key to set.
            value: The value to associate with the key.
        """
        self._data[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the game state to a dictionary.
        
        Returns:
            A dictionary representation of the game state.
        """
        return self._data.copy()
    
    def save_to_json(self, file_path: str) -> None:
        """
        Save the game state to a JSON file.
        
        Args:
            file_path: Path to save the JSON file.
            
        Raises:
            IOError: If there's an error writing to the file.
        """
        try:
            with open(file_path, 'w') as file:
                json.dump(self._data, file, indent=4)
        except IOError:
            raise IOError(f"Error writing to file {file_path}")
    
    def __getattr__(self, name: str) -> Any:
        """
        Access game state properties using dot notation.
        
        Args:
            name: The attribute/key name to access.
            
        Returns:
            The value, converted to DynamicAccessDict if it's a dictionary.
            
        Raises:
            AttributeError: If the key doesn't exist in the game state.
        """
        if name not in self._data:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
        value = self._data[name]
        
        # Convert nested dictionaries to DynamicAccessDict objects
        if isinstance(value, dict):
            return DynamicAccessDict(value)
        # Convert lists of dictionaries to lists of DynamicAccessDict objects
        elif isinstance(value, list):
            return [DynamicAccessDict(item) if isinstance(item, dict) else item for item in value]
        else:
            return value
    
    def __setattr__(self, name: str, value: Any) -> None:
        """
        Set game state properties using dot notation.
        
        Args:
            name: The attribute/key name to set.
            value: The value to set.
        """
        if name == "_data":
            super().__setattr__(name, value)
        else:
            self._data[name] = value
    
    def __getitem__(self, key: str) -> Any:
        """
        Allow dictionary-style access to the game state.
        
        Args:
            key: The key to retrieve.
            
        Returns:
            The value associated with the key.
            
        Raises:
            KeyError: If the key is not found in the game state.
        """
        value = self._data[key]
        if isinstance(value, dict):
            return DynamicAccessDict(value)
        return value
    
    def __setitem__(self, key: str, value: Any) -> None:
        """
        Allow dictionary-style assignment to the game state.
        
        Args:
            key: The key to set.
            value: The value to associate with the key.
        """
        self._data[key] = value
    
    def __contains__(self, key: str) -> bool:
        """
        Allow 'in' operator to check if a key exists in the game state.
        
        Args:
            key: The key to check.
            
        Returns:
            True if the key exists, False otherwise.
        """
        return key in self._data
    
    def __str__(self) -> str:
        """
        Return a string representation of the game state.
        
        Returns:
            A string representation of the game state.
        """
        return json.dumps(self._data, indent=4)
    
    def __repr__(self) -> str:
        """
        Return a representation of the game state.
        
        Returns:
            A representation of the game state.
        """
        return f"GameState({self._data})"
