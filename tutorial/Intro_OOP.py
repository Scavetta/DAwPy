# Intro to OOP

# So far...
myList = list(range(6))
for item in myList:
    print(f"item is {item}")

# instead, we can use Object-oriented programming
# which is based on classes
# Use the class keyword, like def
# Use capital letters
class PrintList:

    def __init__(self, numberlist):
        self.numberlist = numberlist
    
    def print_list(self):
        for item in self.numberlist:
            print(f"item is {item}")

# __init__ is the initialization of the class
# it's not necessary, but anything here
# Gets called when ever you "initialize" an instance.

# e.g. Instantiate it (i.e. create an instance)
A = PrintList(myList)

# Call a method of the instance
A.print_list()

B = PrintList(list(range(6,46)))
B.print_list()
B.numberlist

# Attribute (class variables) == Variable in imperative
# Method (class functions) == Function in imperative

# We don't need an __init__ method:
class PrintList_2:
    
    def print_list(self, numberlist):
        for item in numberlist:
            print(f"item is {item}")

C = PrintList_2()
C.print_list([1,3,5])