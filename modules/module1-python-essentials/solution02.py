"""
Module 1 - Python Essentials
Assignment 2
Student: Salma Areef Syed
Generated: 2025-10-30
This script was exported from the cleaned notebook.
"""

#!/usr/bin/env python
# coding: utf-8

# # Module 1 - Python Essentials
# **Assignment:** 2
# 
# **Student:** Salma Areef Syed
# 
# **Cleaned:** This notebook was cleaned and reformatted for submission.
# 
# **Date cleaned:** 2025-10-30
# 
# ## Objective
# 
# - (Add a short objective description here.)
# 
# ## Contents
# 
# 1. Problem statement
# 2. Code cells (cleaned)
# 3. Results and conclusions
# 
# 

# Assignment 1: Working with Sequences and JSON
# Create a program that:
# •	Reads a JSON file containing a list of students and their grades.
# •	Sorts the students by their average grade.
# •	Compares their scores using Python sequence methods.
# •	Writes the sorted list to a new JSON file.
# •	Include exception handling for malformed files and missing data.
# 

# In[6]:


import json
import os
from statistics import mean

def create_sample_json(file_path):
    """Creates a sample JSON file with student data."""
    sample_data = [
        {"name": "Alice", "grades": [85, 90, 78]},
        {"name": "Bob", "grades": [82, 88, 91]},
        {"name": "Charlie", "grades": [70, 75, 80]}
    ]
    try:
        with open(file_path, 'w') as file:
            json.dump(sample_data, file, indent=4)
        print(f"Sample JSON written to '{file_path}'.")
    except Exception as e:
        print(f"Failed to write sample JSON: {e}")

def read_json_file(file_path):
    """Reads JSON data from a file."""
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            if not isinstance(data, list):
                raise ValueError("JSON root should be a list of students.")
            return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: File '{file_path}' contains invalid JSON.")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []

def calculate_average(student):
    """Calculates the average grade for a student."""
    try:
        grades = student.get("grades", [])
        if not grades:
            raise ValueError("Missing or empty 'grades' list.")
        return mean(grades)
    except Exception as e:
        print(f"Error calculating average for {student.get('name', 'Unknown')}: {e}")
        return 0

def sort_students_by_average(students):
    """Sorts students by average grade in descending order."""
    return sorted(students, key=lambda s: calculate_average(s), reverse=True)

def compare_sequences(student1, student2):
    """Compares two students' grade sequences."""
    grades1 = student1.get("grades", [])
    grades2 = student2.get("grades", [])
    if grades1 > grades2:
        return f"{student1['name']} has a better grade sequence than {student2['name']}."
    elif grades1 < grades2:
        return f"{student2['name']} has a better grade sequence than {student1['name']}."
    else:
        return f"{student1['name']} and {student2['name']} have equal grade sequences."

def write_json_file(file_path, data):
    """Writes JSON data to a file."""
    try:
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
        print(f"Sorted student data written to '{file_path}'.")
    except Exception as e:
        print(f"Error writing to file '{file_path}': {e}")

def main():
    input_file = 'C:/Users/User/AI_Lab_26thApril2025/AI10_SalmaAreefSyed_Assignment_2/students.json'
    output_file = 'C:/Users/User/AI_Lab_26thApril2025/AI10_SalmaAreefSyed_Assignment_2/sorted_students.json'

    # Step 1: Create the sample JSON file
    create_sample_json(input_file)

    # Step 2: Read the data
    students = read_json_file(input_file)
    if not students:
        return

    # Step 3: Sort the students
    sorted_students = sort_students_by_average(students)

    # Step 4: Compare the top two (if available)
    print("Comparison of first two students' grades (if available):")
    if len(sorted_students) >= 2:
        print(compare_sequences(sorted_students[0], sorted_students[1]))

    # Step 5: Write sorted result
    write_json_file(output_file, sorted_students)

if __name__ == "__main__":
    main()


# Logic Summary: Working with Sequences and JSON
# Read JSON File: The program starts by reading a JSON file that contains student data (names and grades). If the file is malformed or missing required data, an exception is raised.
# 
# Calculate Average Grades: It computes the average grade for each student by summing their grades and dividing by the total number of grades.
# 
# Sort Students: The students are sorted based on their average grades using Python's sorting functions. This ensures that students with the highest grades come first.
# 
# Write to JSON: After sorting, the program writes the updated list of students and their grades to a new JSON file.
# 
# Error Handling: The program includes exception handling for invalid file formats, missing data, and other potential issues to ensure smooth execution without crashing.

# Assignment 2: Exception Handling and User-defined Exceptions
# Develop a banking application that:
# •	Allows users to deposit and withdraw money.
# •	Raises and handles exceptions for:
# o	Insufficient funds
# o	Negative input
# o	Incorrect account operations
# •	Uses a user-defined exception class for handling custom banking errors.
# 

# In[7]:


class BankingError(Exception):
    """Base class for user-defined banking exceptions."""
    pass

class InsufficientFundsError(BankingError):
    """Raised when withdrawal exceeds the account balance."""
    def __init__(self, message="Insufficient funds in account."):
        self.message = message
        super().__init__(self.message)

class NegativeInputError(BankingError):
    """Raised when a deposit or withdrawal amount is negative."""
    def __init__(self, message="Amount must be a positive number."):
        self.message = message
        super().__init__(self.message)

class InvalidOperationError(BankingError):
    """Raised when an invalid operation is attempted."""
    def __init__(self, message="Invalid banking operation."):
        self.message = message
        super().__init__(self.message)

class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        if amount < 0:
            raise NegativeInputError("Cannot deposit a negative amount.")
        self.balance += amount
        print(f"Deposited ₹{amount}. New balance: ₹{self.balance}")

    def withdraw(self, amount):
        if amount < 0:
            raise NegativeInputError("Cannot withdraw a negative amount.")
        if amount > self.balance:
            raise InsufficientFundsError(f"Attempted to withdraw ₹{amount}, but only ₹{self.balance} available.")
        self.balance -= amount
        print(f"Withdrew ₹{amount}. New balance: ₹{self.balance}")

    def __str__(self):
        return f"Account holder: {self.owner}, Balance: ₹{self.balance}"

def main():
    account = BankAccount("Salma Areef", 5000)

    print(account)

    try:
        account.deposit(1000)
        account.withdraw(3000)
        account.withdraw(4000)  # This will trigger InsufficientFundsError
    except BankingError as e:
        print(f"Banking error: {e}")

    try:
        account.deposit(-100)  # This will trigger NegativeInputError
    except BankingError as e:
        print(f"Banking error: {e}")

    try:
        operation = "transfer"
        if operation not in ["deposit", "withdraw"]:
            raise InvalidOperationError(f"'{operation}' is not a valid operation.")
    except BankingError as e:
        print(f"Banking error: {e}")

    print(account)

if __name__ == "__main__":
    main()


# This assignment involves creating a banking application with the following features:
# Deposit and Withdraw Money: Allows users to deposit and withdraw funds from their accounts.
# Exception Handling: Implements custom exception handling for:
# Insufficient Funds: Raised when a user tries to withdraw more than the available balance.
# 
# Negative Input: Raised for negative amounts in deposits or withdrawals.
# Incorrect Operations: Raised for invalid operations (e.g., withdrawing from an inactive account).
# User-defined Exceptions: Custom exceptions (NegativeDepositError, NegativeWithdrawalError, InsufficientFundsError) are used to handle specific banking errors.
# 
# Error Handling with try-except: Ensures smooth execution by catching errors and displaying informative messages to the user.
# The application maintains a simple BankAccount class with methods for depositing, withdrawing, and checking the balance. It demonstrates robust error handling by catching and managing various exceptions that might occur during account operations.

# Assignment 3: File Operations and Object-Oriented Design
# Write a class-based program that:
# •	Reads from and writes to a CSV file.
# •	Encapsulates file operations in class methods.
# •	Defines a FileManager class with:
# o	Private variables
# o	Error handling
# o	Method documentation
# •	Demonstrates inheritance for handling different file types (text, CSV).
# 

# In[8]:


import csv
import os

class FileManager:
    """
    Base class for managing file operations.
    """

    def __init__(self, file_path):
        self.__file_path = file_path

    def _get_file_path(self):
        return self.__file_path

    def read(self):
        """
        Method to be overridden in child classes to read from the file.
        """
        raise NotImplementedError("This method must be overridden in subclasses.")

    def write(self, data):
        """
        Method to be overridden in child classes to write to the file.
        """
        raise NotImplementedError("This method must be overridden in subclasses.")

    def file_exists(self):
        """Checks if the file exists."""
        return os.path.isfile(self.__file_path)

class CSVFileManager(FileManager):
    """
    Handles CSV file reading and writing.
    """

    def read(self):
        """
        Reads data from a CSV file and returns it as a list of dictionaries.
        """
        try:
            with open(self._get_file_path(), mode='r', newline='') as file:
                reader = csv.DictReader(file)
                return list(reader)
        except FileNotFoundError:
            print(f"Error: File '{self._get_file_path()}' not found.")
            return []
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return []

    def write(self, data):
        """
        Writes a list of dictionaries to a CSV file.
        """
        if not data or not isinstance(data, list) or not isinstance(data[0], dict):
            print("Invalid data format for writing to CSV.")
            return

        try:
            with open(self._get_file_path(), mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
            print(f"Data successfully written to '{self._get_file_path()}'.")
        except Exception as e:
            print(f"Error writing to CSV file: {e}")

class TextFileManager(FileManager):
    """
    Handles plain text file reading and writing.
    """

    def read(self):
        """
        Reads the content of a text file and returns it as a string.
        """
        try:
            with open(self._get_file_path(), 'r') as file:
                return file.read()
        except FileNotFoundError:
            print(f"Error: File '{self._get_file_path()}' not found.")
            return ""
        except Exception as e:
            print(f"Error reading text file: {e}")
            return ""

    def write(self, content):
        """
        Writes a string to a text file.
        """
        try:
            with open(self._get_file_path(), 'w') as file:
                file.write(content)
            print(f"Text successfully written to '{self._get_file_path()}'.")
        except Exception as e:
            print(f"Error writing to text file: {e}")

def main():
    # CSV File Operations
    csv_path = 'sample_data.csv'
    csv_manager = CSVFileManager(csv_path)

    sample_csv_data = [
        {"Name": "Alice", "Age": "23", "City": "Mumbai"},
        {"Name": "Bob", "Age": "30", "City": "Delhi"}
    ]
    csv_manager.write(sample_csv_data)
    read_csv = csv_manager.read()
    print("CSV File Read:\n", read_csv)

    # Text File Operations
    text_path = 'sample_text.txt'
    text_manager = TextFileManager(text_path)

    sample_text = "This is a sample text file.\nIt contains plain text."
    text_manager.write(sample_text)
    read_text = text_manager.read()
    print("Text File Read:\n", read_text)

if __name__ == "__main__":
    main()


# Object-oriented design using a FileManager base class.
# 
# Inheritance with CSVFileManager and TextFileManager.
# 
# Encapsulation of file operations.
# 
# Error handling and use of private variables.
# 
# Method documentation using docstrings.
# 📁 Output Files:
# sample_data.csv – CSV file with student-like data
# 
# sample_text.txt – Simple plain text content
# 
# 💡 Highlights:
# Private variable: __file_path (encapsulated)
# 
# Inheritance: CSVFileManager and TextFileManager inherit from FileManager
# 
# Polymorphism: .read() and .write() behave differently for CSV and text
# 
# Robust exception handling and validation
# 
# Uses docstrings for method documentation

# Assignment 4: Iterators and Generators
# Build a custom iterable class that:
# •	Implements __iter__() and __next__() to generate Fibonacci numbers up to a limit.
# •	Also includes:
# o	A generator function
# o	A generator expression
# •	Compares performance and code readability between the three approaches.
# 

# In[10]:


import time

class FibonacciIterator:
    """
    Custom iterable class to generate Fibonacci numbers up to a given limit.
    Implements __iter__ and __next__.
    """
    def __init__(self, limit):
        self.limit = limit
        self.a = 0
        self.b = 1
        self.count = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.count >= self.limit:
            raise StopIteration
        if self.count == 0:
            self.count += 1
            return 0
        elif self.count == 1:
            self.count += 1
            return 1
        else:
            self.a, self.b = self.b, self.a + self.b
            self.count += 1
            return self.b

def fibonacci_generator(limit):
    """
    Generator function to yield Fibonacci numbers up to a limit.
    """
    a, b = 0, 1
    for i in range(limit):
        yield a
        a, b = b, a + b

def fibonacci_generator_expr(limit):
    """
    Generator expression version of Fibonacci sequence (uses list slicing for first N numbers).
    Less efficient for large limits, shown here just for syntax comparison.
    """
    def fib_list(n):
        a, b = 0, 1
        result = []
        for _ in range(n):
            result.append(a)
            a, b = b, a + b
        return result
    return (x for x in fib_list(limit))

def benchmark(method_name, iterable):
    start = time.time()
    result = list(iterable)
    end = time.time()
    print(f"{method_name}:")
    print("  → Output:", result)
    print(f"  → Time: {end - start:.6f} seconds\n")

def main():
    limit = 10

    print("=== Fibonacci Sequence using Different Methods ===\n")

    # Custom iterator class
    benchmark("Class Iterator", FibonacciIterator(limit))

    # Generator function
    benchmark("Generator Function", fibonacci_generator(limit))

    # Generator expression
    benchmark("Generator Expression", fibonacci_generator_expr(limit))

if __name__ == "__main__":
    main()


# FibonacciIterator Class: Implements the __iter__ and __next__ methods to create a custom iterator for Fibonacci numbers.
# 
# fibonacci_generator Function: A generator function that yields Fibonacci numbers up to the limit.
# 
# fibonacci_generator_expr Expression: A generator expression version, a concise but less efficient method for generating Fibonacci numbers.
# 
# benchmark Function: Measures and compares the performance of the three approaches.

# Assignment 5: Advanced Class Design and Multiple Inheritance
# Design a role-based access system:
# •	Use classes like User, Admin, Guest, with shared and specific functionality.
# •	Implement multiple inheritance where appropriate.
# •	Use class and instance variables to track user states.
# •	Override methods and demonstrate polymorphism.
# •	Include documentation strings and error-handling features.
# 

# In[11]:


class User:
    """
    Base class for all users in the system.
    Contains common functionalities like login, logout, and tracking user state.
    """
    def __init__(self, username, email):
        self.username = username
        self.email = email
        self._is_logged_in = False

    def login(self):
        """Logs in the user."""
        if not self._is_logged_in:
            self._is_logged_in = True
            print(f"{self.username} has logged in.")
        else:
            print(f"{self.username} is already logged in.")

    def logout(self):
        """Logs out the user."""
        if self._is_logged_in:
            self._is_logged_in = False
            print(f"{self.username} has logged out.")
        else:
            print(f"{self.username} is not logged in.")

    def view_profile(self):
        """Displays basic user profile."""
        if self._is_logged_in:
            print(f"User Profile of {self.username}:")
            print(f"Username: {self.username}")
            print(f"Email: {self.email}")
        else:
            print(f"Please log in first to view {self.username}'s profile.")

    def __str__(self):
        return f"User: {self.username}, Email: {self.email}"

class Admin(User):
    """
    Admin class inheriting from User.
    Has additional capabilities like managing users and viewing all profiles.
    """
    def __init__(self, username, email, admin_level):
        super().__init__(username, email)
        self.admin_level = admin_level  # e.g., 1 for basic admin, 2 for super admin

    def manage_users(self):
        """Admin can manage users."""
        if self._is_logged_in:
            print(f"{self.username} is managing users.")
        else:
            print("Admin must be logged in to manage users.")

    def view_all_profiles(self):
        """Admin can view all user profiles."""
        if self._is_logged_in:
            print(f"{self.username} is viewing all profiles.")
        else:
            print("Admin must be logged in to view all profiles.")

    def __str__(self):
        return f"Admin: {self.username}, Admin Level: {self.admin_level}, Email: {self.email}"

class Guest(User):
    """
    Guest class inheriting from User.
    Limited functionality, mainly for viewing public profiles.
    """
    def __init__(self, username, email):
        super().__init__(username, email)
        self.guest_access_level = "Limited"

    def view_public_profile(self):
        """Guests can only view public profiles."""
        if self._is_logged_in:
            print(f"{self.username} is viewing a public profile.")
        else:
            print(f"Please log in first to view profiles as a guest.")

    def __str__(self):
        return f"Guest: {self.username}, Access Level: {self.guest_access_level}, Email: {self.email}"

class RoleBasedAccess:
    """
    Role-based access system to handle permissions and states for users.
    """
    def __init__(self):
        self.users = []

    def add_user(self, user):
        """Adds a user to the system."""
        if isinstance(user, User):
            self.users.append(user)
            print(f"User {user.username} has been added.")
        else:
            print("Invalid user type. Only instances of User or its subclasses can be added.")

    def authenticate_user(self, username):
        """Simulate user authentication based on username."""
        for user in self.users:
            if user.username == username:
                print(f"Authentication successful for {username}.")
                return user
        print(f"User {username} not found.")
        return None

def main():
    # Create instances of different user roles
    admin = Admin(username="admin1", email="admin1@example.com", admin_level=2)
    guest = Guest(username="guest1", email="guest1@example.com")
    user = User(username="user1", email="user1@example.com")

    # Create RoleBasedAccess system and add users
    access_system = RoleBasedAccess()
    access_system.add_user(admin)
    access_system.add_user(guest)
    access_system.add_user(user)

    # Authenticate and demonstrate functionality
    authenticated_user = access_system.authenticate_user("admin1")
    if authenticated_user:
        authenticated_user.login()
        authenticated_user.view_profile()
        if isinstance(authenticated_user, Admin):
            authenticated_user.manage_users()
            authenticated_user.view_all_profiles()
        authenticated_user.logout()

    authenticated_user = access_system.authenticate_user("guest1")
    if authenticated_user:
        authenticated_user.login()
        authenticated_user.view_public_profile()
        authenticated_user.logout()

    authenticated_user = access_system.authenticate_user("user1")
    if authenticated_user:
        authenticated_user.login()
        authenticated_user.view_profile()
        authenticated_user.logout()

if __name__ == "__main__":
    main()


# 🧩 Breakdown of the System:
# User Class: The base class for all users, with basic functionality like login(), logout(), and view_profile().
# Admin Class: Inherits from User and adds capabilities like managing users and viewing all profiles.
# Guest Class: Also inherits from User and has limited functionality, such as viewing public profiles only.
# RoleBasedAccess Class: Manages users in the system and provides methods like add_user() and authenticate_user() to handle users and their permissions.
# 
# Polymorphism:
# The view_profile() and view_public_profile() methods are overridden by Admin and Guest to provide role-specific behaviors.
# 
# Error Handling:
# Handles login/logout state with basic error messages if actions are attempted while not logged in.
# Checks for valid user types when adding a user to the system.
