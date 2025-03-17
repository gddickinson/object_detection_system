from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
                           QPushButton, QMessageBox, QCheckBox, QGroupBox, QFormLayout)
from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtGui import QIcon, QPixmap
import os
import json
import hashlib
import logging
import time

logger = logging.getLogger('object_detection.auth')

class User:
    """User data container class."""
    def __init__(self, username, role="user", settings=None):
        self.username = username
        self.role = role
        self.settings = settings or {}
        self.login_time = time.time()
    
    @property
    def is_admin(self):
        return self.role == "admin"
    
    def __str__(self):
        return f"User({self.username}, {self.role})"


class LoginWindow(QDialog):
    """Login dialog window."""
    
    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        self.users_file = self.config.get('auth', {}).get('users_file', 'data/users/users.json')
        self.user = None
        self.settings = QSettings("ObjectDetectionApp", "Login")
        
        self.init_ui()
        self.load_saved_username()
        
        # Create default user if no user exists
        self.create_default_user_if_needed()
    
    def init_ui(self):
        """Initialize the login UI."""
        self.setWindowTitle("Login - Object Detection System")
        self.setFixedSize(400, 300)
        
        main_layout = QVBoxLayout()
        
        # App title and logo
        title_layout = QHBoxLayout()
        title_label = QLabel("Object Detection System")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        title_layout.addWidget(title_label)
        main_layout.addLayout(title_layout)
        
        # Login form
        login_group = QGroupBox("Login")
        login_layout = QFormLayout()
        
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Enter username")
        login_layout.addRow("Username:", self.username_input)
        
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setPlaceholderText("Enter password")
        login_layout.addRow("Password:", self.password_input)
        
        self.remember_checkbox = QCheckBox("Remember username")
        login_layout.addRow("", self.remember_checkbox)
        
        login_group.setLayout(login_layout)
        main_layout.addWidget(login_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.login_button = QPushButton("Login")
        self.login_button.clicked.connect(self.verify_login)
        button_layout.addWidget(self.login_button)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)
        
        main_layout.addLayout(button_layout)
        
        # Add a link to create a new user
        create_user_layout = QHBoxLayout()
        create_user_label = QLabel("<a href='#'>Create new user</a>")
        create_user_label.setTextFormat(Qt.RichText)
        create_user_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        create_user_label.linkActivated.connect(self.show_create_user_dialog)
        create_user_layout.addWidget(create_user_label)
        main_layout.addLayout(create_user_layout)
        
        self.setLayout(main_layout)
    
    def load_saved_username(self):
        """Load the saved username if available."""
        saved_username = self.settings.value("username", "")
        if saved_username:
            self.username_input.setText(saved_username)
            self.remember_checkbox.setChecked(True)
            # Focus on password field if username is saved
            self.password_input.setFocus()
        else:
            # Otherwise focus on username field
            self.username_input.setFocus()
    
    def save_username(self, username):
        """Save the username if remember is checked."""
        if self.remember_checkbox.isChecked():
            self.settings.setValue("username", username)
        else:
            self.settings.remove("username")
    
    def verify_login(self):
        """Verify the entered login credentials."""
        username = self.username_input.text().strip()
        password = self.password_input.text()
        
        if not username or not password:
            QMessageBox.warning(self, "Login Failed", "Username and password cannot be empty.")
            return
        
        if self.authenticate_user(username, password):
            # Save username if remember is checked
            self.save_username(username)
            
            # Store the logged-in user info
            self.user = self.get_user_data(username)
            
            logger.info(f"User logged in: {username}")
            self.accept()
        else:
            QMessageBox.warning(self, "Login Failed", "Invalid username or password.")
            self.password_input.clear()
            self.password_input.setFocus()
    
    def get_user(self):
        """Return the logged-in user."""
        return self.user
    
    def authenticate_user(self, username, password):
        """
        Authenticate a user against the stored credentials.
        
        Args:
            username: The username to check
            password: The password to verify
            
        Returns:
            True if authentication succeeded, False otherwise
        """
        users = self.load_users()
        
        if username in users:
            # Get the stored password hash and salt
            stored_hash = users[username].get('password_hash')
            salt = users[username].get('salt')
            
            if stored_hash and salt:
                # Hash the provided password with the stored salt
                password_hash = self.hash_password(password, salt)
                
                # Compare with the stored hash
                return password_hash == stored_hash
        
        return False
    
    def create_default_user_if_needed(self):
        """Create a default admin user if no users exist."""
        users = self.load_users()
        
        if not users:
            # Create the default admin user
            username = "admin"
            password = "admin"  # Default password
            
            salt = os.urandom(32).hex()
            password_hash = self.hash_password(password, salt)
            
            users[username] = {
                'password_hash': password_hash,
                'salt': salt,
                'role': 'admin',
                'created_at': time.time(),
                'settings': {}
            }
            
            # Save the users file
            self.save_users(users)
            
            logger.info("Created default admin user")
            
            # Show info message
            QMessageBox.information(self, "Default User Created", 
                                  "A default admin user has been created.\nUsername: admin\nPassword: admin\n\n"
                                  "Please change the password after logging in.")
    
    def show_create_user_dialog(self):
        """Show dialog to create a new user."""
        dialog = CreateUserDialog(self)
        if dialog.exec_():
            # Refresh the users list
            self.load_users()
            QMessageBox.information(self, "User Created", "New user has been created successfully.")
    
    def load_users(self):
        """Load users from the users file."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.users_file), exist_ok=True)
            
            if os.path.exists(self.users_file):
                with open(self.users_file, 'r') as f:
                    return json.load(f)
            else:
                return {}
        except Exception as e:
            logger.error(f"Error loading users: {e}")
            return {}
    
    def save_users(self, users):
        """Save users to the users file."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.users_file), exist_ok=True)
            
            with open(self.users_file, 'w') as f:
                json.dump(users, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Error saving users: {e}")
            return False
    
    def hash_password(self, password, salt):
        """
        Hash a password with the given salt.
        
        Args:
            password: The password to hash
            salt: The salt to use
            
        Returns:
            The password hash
        """
        # Combine password and salt
        salted_password = password.encode() + bytes.fromhex(salt)
        
        # Create hash
        hash_obj = hashlib.sha256(salted_password)
        
        return hash_obj.hexdigest()
    
    def get_user_data(self, username):
        """
        Get user data for the given username.
        
        Args:
            username: The username to look up
            
        Returns:
            User object
        """
        users = self.load_users()
        
        if username in users:
            user_data = users[username]
            return User(
                username=username,
                role=user_data.get('role', 'user'),
                settings=user_data.get('settings', {})
            )
        
        return None


class CreateUserDialog(QDialog):
    """Dialog for creating a new user."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI."""
        self.setWindowTitle("Create New User")
        self.setFixedSize(300, 200)
        
        layout = QVBoxLayout()
        
        form_layout = QFormLayout()
        
        self.username_input = QLineEdit()
        form_layout.addRow("Username:", self.username_input)
        
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        form_layout.addRow("Password:", self.password_input)
        
        self.confirm_password_input = QLineEdit()
        self.confirm_password_input.setEchoMode(QLineEdit.Password)
        form_layout.addRow("Confirm Password:", self.confirm_password_input)
        
        layout.addLayout(form_layout)
        
        button_layout = QHBoxLayout()
        
        self.create_button = QPushButton("Create")
        self.create_button.clicked.connect(self.create_user)
        button_layout.addWidget(self.create_button)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def create_user(self):
        """Create a new user with the entered credentials."""
        username = self.username_input.text().strip()
        password = self.password_input.text()
        confirm_password = self.confirm_password_input.text()
        
        # Validate input
        if not username:
            QMessageBox.warning(self, "Invalid Input", "Username cannot be empty.")
            return
        
        if not password:
            QMessageBox.warning(self, "Invalid Input", "Password cannot be empty.")
            return
        
        if password != confirm_password:
            QMessageBox.warning(self, "Invalid Input", "Passwords do not match.")
            return
        
        # Load existing users
        users = self.parent.load_users()
        
        # Check if user already exists
        if username in users:
            QMessageBox.warning(self, "Invalid Input", "Username already exists.")
            return
        
        # Create new user
        salt = os.urandom(32).hex()
        password_hash = self.parent.hash_password(password, salt)
        
        users[username] = {
            'password_hash': password_hash,
            'salt': salt,
            'role': 'user',
            'created_at': time.time(),
            'settings': {}
        }
        
        # Save users
        if self.parent.save_users(users):
            self.accept()
        else:
            QMessageBox.critical(self, "Error", "Failed to create user. Please try again.")
