import os
import json
import hashlib
import time
import logging
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
                           QPushButton, QMessageBox, QTableWidget, QTableWidgetItem,
                           QHeaderView, QComboBox, QFormLayout)

logger = logging.getLogger('object_detection.auth.user_manager')

class UserManager:
    """Class for managing users."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.users_file = self.config.get('auth', {}).get('users_file', 'data/users/users.json')
    
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
    
    def create_user(self, username, password, role='user'):
        """
        Create a new user.
        
        Args:
            username: Username for the new user
            password: Password for the new user
            role: User role (default: 'user')
            
        Returns:
            True if the user was created successfully, False otherwise
        """
        # Load existing users
        users = self.load_users()
        
        # Check if user already exists
        if username in users:
            logger.warning(f"User {username} already exists")
            return False
        
        # Create salt and hash password
        salt = os.urandom(32).hex()
        password_hash = self.hash_password(password, salt)
        
        # Create user entry
        users[username] = {
            'password_hash': password_hash,
            'salt': salt,
            'role': role,
            'created_at': time.time(),
            'settings': {}
        }
        
        # Save users
        success = self.save_users(users)
        
        if success:
            logger.info(f"User {username} created successfully")
        else:
            logger.error(f"Failed to create user {username}")
        
        return success
    
    def update_user(self, username, data):
        """
        Update a user.
        
        Args:
            username: Username of the user to update
            data: Dictionary containing the fields to update
            
        Returns:
            True if the user was updated successfully, False otherwise
        """
        # Load existing users
        users = self.load_users()
        
        # Check if user exists
        if username not in users:
            logger.warning(f"User {username} not found")
            return False
        
        # Update user data
        for key, value in data.items():
            if key == 'password':
                # Update password
                salt = os.urandom(32).hex()
                password_hash = self.hash_password(value, salt)
                users[username]['password_hash'] = password_hash
                users[username]['salt'] = salt
            else:
                # Update other fields
                users[username][key] = value
        
        # Save users
        success = self.save_users(users)
        
        if success:
            logger.info(f"User {username} updated successfully")
        else:
            logger.error(f"Failed to update user {username}")
        
        return success
    
    def delete_user(self, username):
        """
        Delete a user.
        
        Args:
            username: Username of the user to delete
            
        Returns:
            True if the user was deleted successfully, False otherwise
        """
        # Load existing users
        users = self.load_users()
        
        # Check if user exists
        if username not in users:
            logger.warning(f"User {username} not found")
            return False
        
        # Delete user
        del users[username]
        
        # Save users
        success = self.save_users(users)
        
        if success:
            logger.info(f"User {username} deleted successfully")
        else:
            logger.error(f"Failed to delete user {username}")
        
        return success
    
    def verify_password(self, username, password):
        """
        Verify a user's password.
        
        Args:
            username: Username to check
            password: Password to verify
            
        Returns:
            True if the password is correct, False otherwise
        """
        # Load users
        users = self.load_users()
        
        # Check if user exists
        if username not in users:
            return False
        
        # Get the stored password hash and salt
        stored_hash = users[username].get('password_hash')
        salt = users[username].get('salt')
        
        if stored_hash and salt:
            # Hash the provided password with the stored salt
            password_hash = self.hash_password(password, salt)
            
            # Compare with the stored hash
            return password_hash == stored_hash
        
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


class UserManagementDialog(QDialog):
    """Dialog for managing users."""
    
    def __init__(self, parent=None, config=None):
        super().__init__(parent)
        self.config = config or {}
        self.user_manager = UserManager(config)
        self.init_ui()
        self.load_users_table()
    
    def init_ui(self):
        """Initialize the UI."""
        self.setWindowTitle("User Management")
        self.setMinimumSize(500, 400)
        
        layout = QVBoxLayout()
        
        # User table
        self.user_table = QTableWidget()
        self.user_table.setColumnCount(4)
        self.user_table.setHorizontalHeaderLabels(["Username", "Role", "Created", "Actions"])
        self.user_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.user_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.user_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.user_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        
        layout.addWidget(self.user_table)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.add_button = QPushButton("Add User")
        self.add_button.clicked.connect(self.show_add_user_dialog)
        button_layout.addWidget(self.add_button)
        
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.load_users_table)
        button_layout.addWidget(self.refresh_button)
        
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        button_layout.addWidget(self.close_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def load_users_table(self):
        """Load users into the table."""
        # Clear the table
        self.user_table.setRowCount(0)
        
        # Load users
        users = self.user_manager.load_users()
        
        # Add users to the table
        for i, (username, data) in enumerate(users.items()):
            self.user_table.insertRow(i)
            
            # Username
            username_item = QTableWidgetItem(username)
            self.user_table.setItem(i, 0, username_item)
            
            # Role
            role_item = QTableWidgetItem(data.get('role', 'user'))
            self.user_table.setItem(i, 1, role_item)
            
            # Created at
            created_at = data.get('created_at', 0)
            created_at_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(created_at))
            created_item = QTableWidgetItem(created_at_str)
            self.user_table.setItem(i, 2, created_item)
            
            # Actions
            actions_widget = QWidget()
            actions_layout = QHBoxLayout()
            actions_layout.setContentsMargins(0, 0, 0, 0)
            
            edit_button = QPushButton("Edit")
            edit_button.clicked.connect(lambda _, u=username: self.show_edit_user_dialog(u))
            actions_layout.addWidget(edit_button)
            
            delete_button = QPushButton("Delete")
            delete_button.clicked.connect(lambda _, u=username: self.delete_user(u))
            actions_layout.addWidget(delete_button)
            
            actions_widget.setLayout(actions_layout)
            self.user_table.setCellWidget(i, 3, actions_widget)
    
    def show_add_user_dialog(self):
        """Show dialog to add a new user."""
        dialog = AddUserDialog(self, self.user_manager)
        if dialog.exec_():
            self.load_users_table()
    
    def show_edit_user_dialog(self, username):
        """Show dialog to edit a user."""
        dialog = EditUserDialog(self, self.user_manager, username)
        if dialog.exec_():
            self.load_users_table()
    
    def delete_user(self, username):
        """Delete a user."""
        # Confirm deletion
        result = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete user '{username}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if result == QMessageBox.Yes:
            # Delete the user
            if self.user_manager.delete_user(username):
                QMessageBox.information(self, "Success", f"User '{username}' deleted successfully.")
                self.load_users_table()
            else:
                QMessageBox.critical(self, "Error", f"Failed to delete user '{username}'.")


class AddUserDialog(QDialog):
    """Dialog for adding a new user."""
    
    def __init__(self, parent=None, user_manager=None):
        super().__init__(parent)
        self.user_manager = user_manager
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI."""
        self.setWindowTitle("Add User")
        self.setFixedSize(300, 200)
        
        layout = QVBoxLayout()
        
        form_layout = QFormLayout()
        
        self.username_input = QLineEdit()
        form_layout.addRow("Username:", self.username_input)
        
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        form_layout.addRow("Password:", self.password_input)
        
        self.role_combo = QComboBox()
        self.role_combo.addItems(["user", "admin"])
        form_layout.addRow("Role:", self.role_combo)
        
        layout.addLayout(form_layout)
        
        button_layout = QHBoxLayout()
        
        self.add_button = QPushButton("Add")
        self.add_button.clicked.connect(self.add_user)
        button_layout.addWidget(self.add_button)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def add_user(self):
        """Add a new user."""
        username = self.username_input.text().strip()
        password = self.password_input.text()
        role = self.role_combo.currentText()
        
        # Validate input
        if not username:
            QMessageBox.warning(self, "Invalid Input", "Username cannot be empty.")
            return
        
        if not password:
            QMessageBox.warning(self, "Invalid Input", "Password cannot be empty.")
            return
        
        # Create the user
        if self.user_manager.create_user(username, password, role):
            QMessageBox.information(self, "Success", f"User '{username}' created successfully.")
            self.accept()
        else:
            QMessageBox.critical(self, "Error", f"Failed to create user '{username}'.")


class EditUserDialog(QDialog):
    """Dialog for editing a user."""
    
    def __init__(self, parent=None, user_manager=None, username=None):
        super().__init__(parent)
        self.user_manager = user_manager
        self.username = username
        self.init_ui()
        self.load_user_data()
    
    def init_ui(self):
        """Initialize the UI."""
        self.setWindowTitle(f"Edit User: {self.username}")
        self.setFixedSize(300, 200)
        
        layout = QVBoxLayout()
        
        form_layout = QFormLayout()
        
        self.password_label = QLabel("Leave blank to keep current password")
        form_layout.addRow("", self.password_label)
        
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        form_layout.addRow("New Password:", self.password_input)
        
        self.role_combo = QComboBox()
        self.role_combo.addItems(["user", "admin"])
        form_layout.addRow("Role:", self.role_combo)
        
        layout.addLayout(form_layout)
        
        button_layout = QHBoxLayout()
        
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_user)
        button_layout.addWidget(self.save_button)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def load_user_data(self):
        """Load user data."""
        users = self.user_manager.load_users()
        
        if self.username in users:
            user_data = users[self.username]
            
            # Set role
            role = user_data.get('role', 'user')
            index = self.role_combo.findText(role)
            if index >= 0:
                self.role_combo.setCurrentIndex(index)
    
    def save_user(self):
        """Save the user."""
        password = self.password_input.text()
        role = self.role_combo.currentText()
        
        # Create update data
        data = {'role': role}
        
        # Add password if provided
        if password:
            data['password'] = password
        
        # Update the user
        if self.user_manager.update_user(self.username, data):
            QMessageBox.information(self, "Success", f"User '{self.username}' updated successfully.")
            self.accept()
        else:
            QMessageBox.critical(self, "Error", f"Failed to update user '{self.username}'.")
