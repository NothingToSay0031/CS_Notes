# Overview

## 01 Intro, Hello World Java

**Our First Java Program.** Printing Hello World is as easy as:

```
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello world!");
    }
}
```

**Key Syntax Features.** Our first programs reveal several important syntax features of Java:

- All code lives inside a class.
- The code that is executed is inside a function, a.k.a. method, called `main`.
- Curly braces are used to denote the beginning and end of a section of code, e.g. a class or method declaration.
- Statements end with semi-colons.
- Variables have declared types, also called their “**static type**”.
- Variables must be declared before use.
- Functions must have a return type. If a function does not return anything, we use void,
- The compiler ensures type consistency. If types are inconsistent, the program will not compile.

**Static Typing.** Static typing is (in my opinion) one  of the best features of Java. It gives us a number of important  advantages over languages without static typing:

- Types are checked before the program is even run, allowing developers to catch type errors with ease.
- If you write a program and distribute the compiled version, it is  (mostly) guaranteed to be free of any type errors. This makes your code  more reliable.
- Every variable, parameter, and function has a declared type,  making it easier for a programmer to understand and reason about code.

There are downside of static typing, to be discussed later.

**Coding Style.** Coding style is very important in 61B  and the real world. Code should be appropriately commented as described in the textbook and lectures.

**Command line compilation and execution.** `javac` is used to compile programs. `java` is used to execute programs. We must always compile before execution.

## 02 Defining and Using Classes

**Client Programs and Main Methods.** A Java program without a main method cannot be run using the `java` command. However, methods from one class can be invoked using the `main` method of another class. A java class that uses another class is called **a client of that class**.

**Class Declaration.** Java classes can contain methods and/or variables. We say that such methods and variables are “members” of the calss. Members can be ***instance*** members or ***static*** members. Static members are declared with the `static` keyword. Instance members are any members without the `static` keyword.

**Class Instantiation.** Instantiating a class is almost always done using the **new** keyword, e.g. `Dog d = new Dog()`. An instance of a class in Java is also called an `Object`.

**Dot Notation.** We access members of a class using dot notation, e.g. `d.bark()`. Class members can be accessed from within the same class or from other classes.

**Constructors.** Constructors tell Java what to do when a program tries to create an instance of a class, e.g. what it should do when it executes `Dog d = new Dog()`.

**Array Instantiation.** Arrays are also instantiated using the `new` keyword. If we have an array of Objects, e.g. `Dog[] dogarray`, then each element of the array must also be instantiated separately.

```
Dog[] dogs = new Dog[2];
dogs[0] = new Dog(8);
```

**Static vs. Instance methods.** The distinction between static and instance methods is incredibly important. Instance methods are actions that can only be taken by an instance of the class (i.e. a specific object), whereas static methods are taken by the class itself. An instance method is invoked using a reference to a specific instance, e.g. `d.bark()`, whereas static methods should be invoked using the class name, e.g. `Math.sqrt()`. Know when to use each.

**Static variables.** Variables can also be static. Static variables should be accessed using the class name, e.g. `Dog.binomen` as opposed to `d.binomen`. Technically Java allows you to access using a specific instance, but we strongly encourage you not to do this to avoid confusion.

**void methods.** A method which does not return anything should be given a void return type.

**The `this` keyword.** Inside a method, we can use the `this` keyword to refer to the current instance.

**public static void main(String[] args).** We now know what each of these things means:

- public: So far, all of our methods start with this keyword.
- static: It is a static method, not associated with any particular instance.
- void: It has no return type.
- main: This is the name of the method.
- String[] args: This is a parameter that is passed to the main method.

**Command Line Arguments.** Arguments can be provided by the operating system to your program as “command line arguments,” and can be accessed using the `args` parameter in `main`. For example if we call our program from the command line like this `java ArgsDemo these are command line arguments`, then the `main` method of `ArgsDemo` will have an array containing the Strings “these”, “are”, “command”, “line”, and “arguments”.

**Using Libraries.** There’s no need in the year 2021 to build everything yourself from scratch. In our course, you are allowed to and highly encouraged to use Java’s built-in libraries, as well as libraries that we provide, e.g. the Princeton standard library. You should not use libraries other than those provided or built into Java because it may render some of the assignments moot, and also our autograder won’t have access to these libraries and your code won’t work.

**Getting Help from the Internet.** You’re welcome to seek help online. However, you should always cite your sources, and you should not seek help on specific homework problems or projects. For example, googling “how convert String Java” or “how read file Java” are fine, but you should not be searching “project 2 61b java berkeley”.

## 03 References, Recursion, and Lists

**Bits** The computer stores information as memory, and represents this information using sequences of bits, which are either 0 or 1.

**Primitives** Primitives are representations of information. There are 8 primitive types in Java: **byte, short, int, long, float, double, boolean, and char.** Each primitive is represented by a certain number of bits. For example, ints are 32 bit primitives, while bytes are 8 bit primitives.

**Declaring Primitives** When we declare a variable to be a primitive (i.e. `int x;`), we set aside enough memory space to hold the bits (in this case, 32). We can think of this as a box holding the bits. Java then maps the variable name to this box. Say we have a line of code `int y = x;` where `x` was defined before. Java will copy the bits inside the `x` box into the bits in the `y` box.

**Creating Objects** When we create an instance of a class using the `new` keyword, Java creates boxes of bits for each field, where the size of each box is defined by the type of each field. For example, if a Walrus object has an `int` variable and a `double` variable, then Java will allocate two boxes totaling 96 bits (32 + 64) to hold both variables. These will be set to a default value like 0. The constructor then comes in and fills in these bits to their appropriate values. The return value of the constructor will return the location in memory where the boxes live, usually an address of 64 bits. This address can then be stored in a variable with a “reference type.”

**Reference Types** <u>If a variable is not a primitive type, then it is a reference type.</u> When we declare object variables, we use reference type variables to store the location in memory of where an object is located. Remember this is what the constructor returns. A reference type is always a box of size 64 bits. Note that the variable does not store the entire object itself!

**Golden Rule of Equals** For primitives, the line `int y = x` copies the bits inside the `x` box into the `y` box. For reference types, we do the exact same thing. In the line `Walrus newWalrus = oldWalrus;`, we copy the 64 bit address in the `oldWalrus` box into the `newWalrus` box. So we can think of this golden rule of equals (GroE) as: when we assign a value with equals, we are just copying the bits from one memory box to another!

**Parameter Passing** Say we have a method `average(double a, double b)`. This method takes two doubles as parameters. Parameter passing also follows the GRoE, i.e. when we call this method and pass in two doubles, we copy the bits from those variables into the parameter variables.

**Array Instantiation.** Arrays are also Objects, and are also instantiated using the `new` keyword. This means declaring an array variable (i.e. `int[] x;`) will create a 64-bit reference type variable that will hold the location of this array. Of course, right now, this box contains the value null, as we have not created the array yet. The `new` keyword for arrays will create the array and return the location of this array in memory. So by saying `int[] x = new int[]{0, 1, 2, 3, 4};`, we set the location of this newly created array to the variable x. Note that the size of the array was specified when the array was created, and cannot be changed!

**IntLists.** Using references, we recursively defined the `IntList` class. `IntLists` are lists of integers that can change size (unlike arrays), and store an arbitrarily large number of integers. Writing a `size` helper method can be done with either recursion or iteration.

## 04 SLLists, Nested Classes, Sentinel Nodes

**Naked Data Structures** `IntLists` are hard to use. In order to use an `IntList` correctly, the programmer must understand and utilize recursion even for simple list related tasks.

**Adding Clothes** First, we will turn the `IntList` class into an `IntNode` class. Then, we will delete any methods in the `IntNode` class. Next, we will create a new class called `SLList`, which contains the instance variable `first`, and this variable should be of type `IntNode`. In essence, we have “wrapped” our `IntNode` with an `SLList`.

**Using SLList** As a user, to create a list, I call the constructor for `SLList`, and pass in the number I wish to fill my list with. The `SLList` constructor will then call the `IntList` constructor with that number, and set `first` to point to the `IntList` it just created.

**Improvement** Notice that when creating a list with one value, we wrote `SLList list = new SLList(1)`. We did not have to worry about passing in a null value like we did with our `IntList`. Essentially, the SLList class acts as a middleman between the list user and the naked `IntList`.

**Public vs. Private** We want users to modify our list via `SLList` methods only, and not by directly modifying `first`. We can prevent other users from doing so by setting our variable access to `private`. Writing `private IntNode first;` prevents code in other classes from accessing and modifying `first` (while the code inside the class can still do so).

**Nested Classes** We can also move classes into classes to make nested classes! You can also declare the nested classes to be private as well; this way, other classes can never use this nested class.

**Static Nested Classes** If the `IntNode` class never uses any variable or method of the `SLList` class, we can turn this class static by adding the “static” keyword.

**Recursive Helper Methods** If we want to write a recursive method in `SLList`, how would we go about doing that? After all, the `SLList` is not a naturally recursive data structure like the `IntNode`. A common idea is to write an outer method that users can call. This method calls a private helper method that takes `IntNode` as a parameter. This helper method will then perform the recursion, and return the answer back to the outer method.

**Caching** Previously, we calculated the size of our `IntList` recursively by returning 1 + the size of the rest of our list. This becomes really slow if our list becomes really big, and we repeatedly call our size method. Now that we have an `SLList`, lets simply cache the size of our list as an instance variable! Note that we could not do this before with out `IntList`.

**Empty Lists** With an`SLList`, we can now represent an empty list. We simply set `first` to `null` and `size` to `0`. However, we have introduced some bugs; namely, because `first` is now `null`, any method that tries to access a property of `first` (like `first.item`) will return a `NullPointerException`. Of course, we can fix this bug by writing code that handles this special case. But there may be many special cases. Is there a better solution?

**Sentinel Nodes** Lets make all `SLList` objects, even empty lists, the same. To do this, lets give each SLList a sentinel node, a node that is always there. Actual elements go after the sentinel node, and all of our methods should respect the idea that sentinel is always the first element in our list.

**Invariants** An invariant is a fact about a data structure that is guaranteed to be true (assuming there are no bugs in your code). This gives us a convenient checklist every time we add a feature to our data structure. Users are also guaranteed certain properties that they trust will be maintained. For example, an `SLList` with a sentinel node has at least the following invariants:

- The sentinel reference always points to a sentinel node.
- The front item (if it exists), is always at sentinel.next.item.
- The size variable is always the total number of items that have been added.

## 05 DLLists, Arrays

**SLList Drawbacks** `addLast()` is slow! We can’t add to the middle of our list. In addition, if our list is really large, we have to start at the front, and loop all the way to the back of our list before adding our element.

**A Naive Solution** Recall that we cached the size of our list as an instance variable of `SLList`. What if we cached the `last` element in our list as well? All of a sudden, `addLast()` is fast again; we access the last element immediately, then add our element in. But `removeLast()` is still slow. In `removeLast()`, we have to know what our second-to-last element is, so we can point our cached `last` variable to it. We could then cache a `second-to-last` variable, but now if I ever want to remove the second-to-last element, I need to know where our third-to-last element is. How to solve this problem?

**DLList** The solution is to give each `IntNode` a `prev` pointer, pointing to the previous item. This creates a doubly-linked list, or `DLList`. With this modification, adding and removing from the front and back of our list becomes fast (although adding/removing from the middle remains slow).

**Incorporating the Sentinel** Recall that we added a sentinel node to our `SLList`. For `DLList`, we can either have two sentinels (one for the front, and one for the back), or we can use a circular sentinel. A `DLList` using a circular sentinel has one sentinel. The sentinel points to the first element of the list with `next` and the last element of the list with `prev`. In addition, the last element of the list’s `next` points to the sentinel and the first element of the list’s `prev` points to the sentinel. For an empty list, the sentinel points to itself in both directions.

**Generic DLList** How can we modify our `DLList` so that it can be a list of whatever objects we choose? Recall that our class definition looks like this:

```
public class DLList { ... }
```

We will change this to

```
public class DLList<T> { ... }
```

where `T` is a placeholder object type. Notice the angle bracket syntax. Also note that we don’t have to use `T`; any variable name is fine. In our `DLList`, our item is now of type `T`, and our methods now take `T` instances as parameters. We can also rename our `IntNode` class to `TNode` for accuracy.

**Using Generic DLList** Recall that to create a `DLList`, we typed:

```
DLList list = new DLList(10);
```

If we now want to create a `DLList` holding `String` objects, then we must say:

```
DLList<String> list = new DLList<>("bone");
```

On list creation, the compiler replaces all instances of `T` with `String`! We will cover generic typing in more detail in later lectures.

**Arrays** Recall that variables are just boxes of bits. For example, `int x;` gives us a memory box of 32 bits. Arrays are a special object which consists of a numbered sequence of memory boxes! To get the ith item of array `A`, use `A[i]`. The length of an array cannot change, and all the elements of the array must be of the same type (this is different from a Python list). The boxes are zero-indexed, meaning that for a list with N elements, the first element is at `A[0]` and the last element is at `A[N - 1]`. Unlike regular classes, **arrays do not have methods!** Arrays do have a `length` variable though.

**Instantiating Arrays** There are three valid notations for creating arrays. The first way specifies the size of the array, and fills the array with default values:

```
int[] y = new int[3];
```

The second and third ways fill up the array with specific values.

```
int[] x = new int[]{1, 2, 3, 4, 5};
int[] w = {1, 2, 3, 4, 5};
```

We can set a value in an array by using array indexing. For example, we can say `A[3] = 4;`. This will access the **fourth** element of array `A` and sets the value at that box to 4.

**Arraycopy** In order to make a copy of an array, we can use `System.arraycopy`. It takes 5 parameters; the syntax is hard to memorize, so we suggest using various references online such as [this](https://www.tutorialspoint.com/java/lang/system_arraycopy.htm).

**2D Arrays** We can declare multidimensional arrays. For 2D integer arrays, we use the syntax:

```
int[][] array = new int[4][];
```

This creates an array that holds integer arrays. Note that we have to manually create the inner arrays like follows:

```
array[0] = new int[]{0, 1, 2, 3};
```

Java can also create multidemensional arrays with the inner arrays created automatically. To do this, use the syntax:

```
int[][] array = new int[4][4];
```

We can also use the notation:

```
int[][] array = new int[][]{{1}, {1, 2}, {1, 2, 3}}
```

to get arrays with specific values.

**Arrays vs. Classes**

- Both are used to organize a bunch of memory.
- Both have a **fixed number** of “boxes”.
- Arrays are accessed via square bracket notation. Classes are accessed via dot notation.
- **Elements in the array must be all be the same type. Elements in a class may be of different types.**
- **Array indices are computed at runtime. We cannot compute class member variable names.**

## 06 ALists, Resizing, vs. SLists

**Lists vs. Arrays** Our `DLList` has a drawback. Getting the ith item is slow; we have to scan through each item in the list, starting from the beginning or the end, until we reach the ith item. For an array named `A`, however, we can quickly access the ith item using bracket notation, `A[i]`. Thus, our goal is to implement a list with an array.

**AList** The `AList` will have the same API has our `DLList`, meaning it will have the same methods as `DLList` (`addLast()`, `getLast()`, `removeLast()`, and `get(int i)`). The `AList` will also have a `size` variable that tracks its size.

**AList Invariants** There are a few invariants for our `AList`.

- `addLast`: The next item we want to add, will go into position `size`.
- `getLast`: The item we want to return is in position `size - 1`.
- `size`: The number of items in the list should be `size`.

**Implementing AList** Each `AList` has an `int[]` called `items`.

- For `addLast`, we place our item in `items[size]`.
- For `getLast`, we simply return `items[size - 1]`.
- For `removeLast`, we simply decrement `size` (we don’t need to change `items`). Thus, if `addLast` is called next, it simply overwrites the old value, because size was decremented. **However, it is good practice to null out objects when they are removed, as this will save memory.** Notice how closely these methods were related to the invariants.

**Abstraction** One key idea of this course is that the implementation details can be hidden away from the users. For example, a user may want to use a list, but we, as implementers, can give them any implementation of a list, as long as it meets their specifications. A user should have no knowledge of the inner workings of our list.

**Array Resizing** When the array gets too full, we can resize the array. However, we have learned that array size cannot change. The solution is, instead, to create a new array of a larger size, then copy our old array values to the new array. Now, we have all of our old values, but we have more space to add items.

**Resizing Speed** In the lecture video, we started off resizing the array by one more each time we hit our array size limit. This turns out to be extremely slow, because copying the array over to the new array means we have to perform the copy operation for each item. The worst part is, since we only resized by one extra box, if we choose to add another item, we have to do this again each time we add to the array.

**Improving Resize Performance** Instead of adding by an extra box, we can instead create a new array with `size * FACTOR` items, where `FACTOR` could be any number, like 2 for example. We will discuss why this is fast later in the course.

**Downsizing Array Size** What happens if we have a 1 million length array, but we remove 990,000 elements of the array? Well, similarly, we can downsize our array by creating an array of half the size, if we reach 250,000 elements, for example. Again, we will discuss this more rigorously later in the course.

**Aside: Breaking Code Up** Sometimes, we write large methods that do multiple things. A better way is to break our large methods up into many smaller methods. One advantage of this is that we can test our code in parts.

**Generic AList** Last time, we discussed how to make a generic `DLList`. We can do something similar for `AList`. But we find that we error out on array creation. Our problem is that generic arrays are not allowed in Java. Instead, we will change the line:

```
items = new Item[100];
```

to:

```
items = (Item[]) new Object[100];
```

This is called a cast, and we will learn about it in the future.

## 07 Testing

**Why Test Code?** In the real world chances are you won’t have an autograder. When your code gets deployed into production, it is important that you know that it will work for simple cases as well as strange edge cases.

**Test-Driven Development** When provided an autograder, it is very easy to go “autograder happy”. Instead of actually understanding the spec and the requirements for a project, a student may write some base implementation, smash their code against the autograder, fix some parts, and repeat until a test is passed. This process tends to be a bit lengthy and really is not the best use of time. We will introduce a new programming method, Test-Driven Development (TDD) where the programmer writes the tests for a function  BEFORE the actual function is written. Since unit tests are written before the function is, it becomes much easier to isolate errors in your code. Additionally, writing unit test requires that you have a relatively solid understanding of the task that you are undertaking. A drawback of this method is that it can be fairly slow and also sometimes it can be easy to forget to test how functions interact with each other.

**JUnit Tests** JUnit is a package that can is used to debug programs in Java. An example function that comes from JUnit is `assertEquals(expected, actual)`. This function asserts true if expected and actual have the same value and false otherwise. There are a bunch of other JUnit functions such as `assertEquals`, `assertFalse`, and `assertNotNull`.

When writing JUnit tests,  it is good practice to write ‘@Test’ above the functionthat is testing. This allows for all your test methods to be run non statically.

## 08 Inheritance, Implements

**Method Overloading** In Java, methods in a class can have the same name, but different parameters. For example, a `Math` class can have an `add(int a, int b)` method and an `add(float a, float b)` method as well. The Java compiler is smart enough to choose the correct method depending on the parameters that you pass in. **Methods with the same name but different parameters are said to be overloaded.**

**Making Code General** Consider a `largestNumber` method that only takes an AList as a parameter. The drawback is that the logic for `largestNumber` is the same regardless of if we take an `AList` or `SLList`. We just operate on a different type of list. If we use our previous idea of method overriding, we result in a very long Java file with many similar methods. This code is hard to maintain; if we fix a bug in one method, we have to duplicate this fix manually to all the other methods.

The solution to the above problem is to define a new reference type that represents both `AList` and `SLList`. We will call it a `List`. Next, we specify an “is-a” relationship: An `AList` is a `List`. We do the same for `SLList`. Let’s formalize this into code.

**Interfaces** We will use the keyword `interface` instead of `class` to create our `List`. More explicitly, we write:

```
public interface List<Item> { ... }
```

The key idea is that interfaces specify what this `List` can do, not how to do it. Since all lists have a `get` method, we add the following method signature to the interface class:

```
public Item get(int i);
```

Notice we did not define this method. We simply stated that this method should exist as long as we are working with a `List` interface.

Now, we want to specify that an `AList` is a `List`. We will change our class declaration of `AList` to:

```
public AList<Item> implements List<Item> { ... }
```

We can do the same for `SLList`. Now, going back to our `largestNumber` method, instead of creating one method for each type of list, we can simply create one method that takes in a `List`. As long as our actual object implements the `List` interface, then this method will work properly!

**Overriding** For each method in `AList` that we also defined in `List`, we will add an @Override right above the method signature. As an example:

```
@Override
public Item get(int i) { ... }
```

This is not necessary, but is good style and thus we will require it. Also, it allows us to check against typos. If we mistype our method name, the compiler will prevent our compilation if we have the @Override tag.

**Interface Inheritance** Formally, we say that subclasses inherit from the superclass. Interfaces contain all the method signatures, and each subclass must implement every single signature; think of it as a contract. In addition, relationships can span multiple generations. For example, C can inherit from B, which can inherit from A.

**Default Methods** Interfaces can have default methods. We define this via:

```
default public void method() { ... }
```

We can actually implement these methods inside the interface. Note that there are no instance variables to use, but we can freely use the methods that are defined in the interface, without worrying about the implementation. Default methods should work for any type of object that implements the interface! The subclasses do not have to re-implement the default method anywhere; they can simply call it for free. However, we can still override default methods, and re-define the method in our subclass.

**Static vs. Dynamic Type** Every variable in Java has a static type. This is the type specified when the variable is declared, and is checked at compile time. Every variable also has a dynamic type; this type is specified when the variable is instantiated, and is checked at runtime. As an example:

```
Thing a;
a = new Fox();
```

Here, `Thing` is the static type, and `Fox` is the dynamic type. This is fine because all foxes are things. We can also do:

```
Animal b = a;
```

This is fine, because all foxes are animals too. We can do:

```
Fox c = b;
```

This is fine, because `b` points to a `Fox`. Finally, we can do:

```
a = new Squid()
```

This is fine, because the static type of `a` is a `Thing`, and `Squid` is a thing.

**Dynamic Method Selection** The rule is, if we have a static type `X`, and a dynamic type `Y`, then if `Y` overrides the method from `X`, then on runtime, we use the method in `Y` instead. Student often confuse overloading and overriding.

**Overloading and Dynamic Method Selection** Dynamic method selection plays no role when it comes to overloaded methods. Consider the following piece of code, where `Fox extends Animal`.

```
1  Fox f = new Fox();
2  Animal a = f;
3  define(f);
4  define(a);
```

Let’s assume we have the following overloaded methods in the same class:

```
public static void define(Fox f) { ... }
public static void define(Animal a) { ... }
```

Line 3 will execute `define(Fox f)`, while line 4 will execute `define(Animal a)`. Dynamic method selection only applies when we have overridden methods. There is no overriding here, and therefore dynamic method selection does not apply.

## 09 Extends, Casting, Higher Order Functions

**The Interface and implements.** Earlier we went classes and interfaces and we realized that when writing classes, we can sometimes write a lot of redundant code. This leads us to Inheritance, the idea that some object does not need to redefine all of its qualities of its parent. We can inherit from both interfaces and classes and the syntax is slightly different. For classes to inherit the qualities of an interface the syntax is as follows (where SLList is a class and List61B is an interface):

```
SLList<Blorp> implements List61B<Blorp>
```

Similarly, the way for a class to implement the qualities of another class the syntax is as follows:

```
Class_Name extends Class_Name
```

**Usage of Inheritance.** Say we wanted to make a special type of `SLList` called `RotatingSLList`. `RotatingSLList` should be able to do everyhthing that SLList can; however, it should also be able to rotate to the right. How can we do this? Well this is just an application of Inheritance! Doing the following will allow for RotatingSLList to have all the qualities of SLList as well as its own method `rotateRight`.

```
public class RotatingSLList<Blorp> extends SLList<Blorp>{
  public void rotateRight() {...}
}
```

**What is Inherited?** We have a powerful tool in Inheritance now; however, we will define a few rules. For now, we will say that we can inherit:

- instance and static variables
- all methods
- all nested classes 

This changes a little bit with the introduction of private variables but don’t worry about that right now. The one item that is not inherited is a class’s constructor.

**The Special Case of the Constructor?** Even though constructor’s are not inherited, we still use them. We can call the constructor explicitly by using the keyword `super()`. At the start of every constructor, there is already an implicit call to its super class`s constructor. As a result

```
public VengefulSLList() {
  deletedItems = new SLList<Item>();
}
```

is equivalent to

```
public VengefulSLList() {
  super();
  deletedItems = new SLList<Item>();
}
```

However, constructor`s with arguments are not implicitly called. This means that.

```
public VengefulSLList() {
    super(x);
    deletedItems = new SLList<Item>();
  }
```

is not equivalent to

```
public VengefulSLList() {
    deletedItems = new SLList<Item>();
  }
```

This is because only the empty argument `super()` is called.

**Is A.**  When a class inherits from another, we know that it must have all the qualities of it. This means that `VengefulSLList` is a `SLList` because it has all the qualities of an `SLList`- it just has a few additional ones too.

Every single class is a descendent on the Object class, meaning they are all Objects.

**Abstraction** As you’ll learn later in this class, programs can get a tad confusing when they are really large. A way to make programs easier to handle is to use abstraction. Basically abstraction is hiding components of programs that people do not need to see. The user of the hidden methods should be able to use them without knowing how they work.

An intuitive way to realize the motivation of abstraction is to look at yourself. You are a human (unless some robot is looking at this in which case I am sorry for offending you) and humans can eat food and convert it to energy. You do not need to know how you convert food to energy you just know that it works. In this case think of your conversion of food to energy as a method and the input is food and the output is energy.

**Casting** In Java, every object has a static type (defined at compile-time) and a dynamic type (defined at run-time). Our code may rely on the fact that some variable may be a more specific type than the static type. For example if we had the below definitions:

```
Poodle frank  = new Poodle("Frank", 5);
Poodle frankJr = new Poodle("Frank Jr.", 15);
```

This statement would be valid

```
Dog largerDog = maxDog(frank, frankJr);
```

But this one would not be

```
Poodle largerPoodle = maxDog(frank, frankJr);
```

The reason the former statement is valid is because the compilers knows for a fact that anything that is returned from a `maxDog` function call is a `Dog`. However, in the latter case, the compiler does not know for a fact that the return value of `maxDog` would result in a `Poodle` even though both `Dog` arguments are `Poodle`s.

Instead of being happy with just having a generic `Dog`, we can be a bit risky and use a technique called casting. Casting allows us to force the static type of a variable, basically tricking the compiler into letting us force the static type of an expression. To make `largerPoodle` into a static type `Poodle` we will use the following:

```
Poodle largerPoodle = (Poodle) maxDog(frank, frankJr);
```

Note that we are not changing the actual dynamic type of maxDog- we are just telling the compiler what is coming out of maxDog will be a `Poodle`. This means that any reference to `largerPoodle` will have a static type of `Poodle` associated with it.

Casting, while powerful is also quite dangerous. You need to ensure that what you are casting to can and will actually happen. There are a few rules that can be used:

- You can always cast up (to a more generic version of a class) without fear of ruining anything because we know the more specific version is a version of the generic class. For example you can always cast a Poodle to a Dog because all Poodles are Dog’s.
- You can also cast down (to a more specific version of a class) with caution as you need to make sure that, during runtime, nothing is passed in that violates your cast. For example, sometimes Dog’s are Poodle’s but not always.
- Finally, you cannot ever cast to a class that is above or below the class being cast. For an example, you cannot cast a Dog to a Monkey because a Monkey is not in the direct lineage of a Dog- it is a child of animal so a bit more distant.

## 10 Subtype Polymorphism vs. HoFs

**Review: Typing Rules**

- Compiler allows the memory box to hold any subtype.
- Compiler allows calls based on static type.
- Overridden non-static methods are selected at runtime based on dynamic type.
- For overloaded methods, the method is selected at compile time.

**Subtype Polymorphism** Consider a variable of static type `Deque`. The behavior of calling `deque.method()` depends on the dynamic type. Thus, we could have many subclasses the implement the `Deque` interface, all of which will be able to call `deque.method()`.

**Subtype Polymorphism Example** Suppose we want to write a function `max()` that returns the max of any array regardless of type. If we write a method `max(Object[] items)`, where we use the ‘>’ operator to compare each element in the array, this will not work! Why is this the case?

Well, this makes the assumption that all objects can be compared. But some objects cannot! Alternatively, we could write a `max()` function inside the Dog class, but now we have to write a `max()` function for each class that we want to compare! Remember, our goal is to write a “one true max method” that works for all comparable objects.

**Solution: OurComparable Interface** The solution is to create an interface that contains a `compareTo(Object)` method; let’s call this interface `OurComparable`. Now, our `max()` method can take a `OurComparable[]` parameter, and since we guarantee that any object which extends the interface has all the  methods inside the interface, we guarantee that we will always be able to call a `compareTo` method, and that this method will correctly return some ordering of the objects.

Now, we can specify a “one true max method”. Of course, any object that needs to be compared must implement the `compareTo` method. However, instead of re-implementing the `max` logic in every class, we only need to implement the logic for picking the ordering of the objects, given two objects.

**Even Better: Java’s In-Built Comparable** Java has an in-built `Comparable` interface that uses generics to avoid any weird casting issues. Plus, Comparable already works for things like `Integer`, `Character`, and `String`;  moreover, these objects have already implemented a `max`, `min`, etc. method for you. Thus you do not need to re-do work that’s already been done!

**Comparators** The term “Natural Order” is used to refer to the ordering implied by a `Comparable`’s `compareTo` method. However, what if we want to order our `Dog` objects by something other than `size`? We will instead pass in a `Comparator<T>` interface, which demands a `compare()` method. We can then implement the `compare()` method anyway we want to achieve our ordering.

## 11 Libraries, Abstract Classes, Packages

**Abstract Data Types** Previously, we went over interfaces which, in a traditional sense (disregarding default methods which will be defined a bit lower), requires certain methods to be implemented in a class if it is said a type of that interface. Abstract Data Types follow this philosophy, and are defined to be some sort of Object that is defined by some set of operations rather than the implementation of these operations.

**Interfaces** There are 2 types of inheritance that we have gone over in previous lectures:

- Interface inheritance: What (the class can do).
- Implementation inheritance: How (the class does it).

**Default Methods** The way we have dealt with interfaces, there is no content in them. We only define a certain set of operations that need to be fulfilled by anything that implements the interface. However, we can create `default` methods that take the following form:

```
default void methodName(){...}
```

Normal interface methods have no body and only state what needs to be defined. Default methods on the other hand provide how a method by providing a method body.

Here are some bullet points about interfaces

- **variables can exist in interfaces but they are public static final.**
- classes can extend more than 1 interface.
- methods are public unless stated otherwise
- interfaces cannot be instantiated.

**Abstract Classes** Abstract classes can be thought of as a hybrid of a normal class and an interface. Abstract classes are like interfaces in that they cannot be instantiated. All methods in an Abstract class are like normal methods in classes unless they have word `abstract` in front of them. If that is the case then they are treated like normal methods in interfaces and do not need to have a body and can instead have an implementation in whatever class extends them. A very important difference between abstract classes and interfaces is that a class can only extend one abstract class but can implement more than one interface.

**Packages** A namespace is a region that can be used to organize code. Packages are a specific type of namespace that is used to organize classes and interfaces. To use a class from a different package use the  following syntax:

```
package_name.classname.subclassname a = new package_name.classname.subclassname();
```

To make your life easier while typing out code, you can simply import the class following the syntax below:

```
import package_name.classname.subclassname;
```

Replace the subclassname with a * if you want to important everything from the class.

## 12 Generics, Autoboxing

**Autoboxing and Unboxing** Autoboxing is the Java’s automatic conversion of between wrappers (Integer) to primitives (int). In most cases, if Java expects a wrapper class and gets a primitive instead, it autoboxes the primitive. Alternatively, if Java expects a primitive and gets a wrapper, it unboxes the wrapper.

**Drawbacks of Autoboxing and Unboxing** Though you can almost always interchange there are some things to the process.

- Autoboxing and unboxing can cause your program to slow down if you use it too much
- Wrappers require a lot more memory than primitives.
- If an array expects a wrapper and gets a primitive or vice versa, it will error. As in you cannot pass ints into an array whose type is Integer[] or the other way around.

**Immutability** Immutable data types are types that cannot change. To make sure that a variable does not change, use the `final` keyword. Once a variable is declared final, it can never change after initial assignment. An important note is that if an address is declared final it means that the address can’t change- it says nothing about its contents. For example the below syntax is valid:

```
final int[] arr = new int[1];
arr[0] = 1;
arr[0] = 3
```

But this one is not:

```
final int[] arr = new int[3];
arr = new int[4];
```

Because you are changing the address of the actual array the variable is pointing to.

**Generic Classes** To make it so that a class can have variables or methods that have a generic type, use the following syntax:

```
public class ArrayMap<K,V>{...}
```

Then when instantiating the class pass in some “real”, or known, types to the class

**Generic Methods** You can define a method that takes in generic parameters with the following syntax.

```
public static <Chow, Der> Chow get(ArrayMap<Chow, Der)> am, Chow key){...}
```

From left to right we have the declaration of the generics being used in this function then we have the return type. Finally, we have our arguments, the first being an ArrayMap with 2 generics and the latter being a generic type object.

To use a generic method use the following syntax

```
ArrayMap<Integer, String> ismap = new ArrayMap<Integer, String>();
System.out.println(MapHelper.get(ismap, 5));
```

**Comparing Objects with Generic Methods** Now we have the ability to put vague Objects into methods. However this lends itself to a bit of a problem- how do we compare these Objects? We cannot simply use ‘>’ because we aren’t sure if our object is a numerical primitive. We can get around this by using `.compareTo(Object O)`.

Now we have a new problem. How do we know if our generic has a compareTo method. To get around this, we can make sure that our generic must be a type of our `OurComparable`. How do we do this? Well take a gander below and check it out.

```
public static <K extends OurComparable, V> K maxKey(ArrayMap<K, V> am) {
  ...
  if (k.compareTo(largest) > 0) {
    ...
}
```

Basically what’s happening is that, in the header, we ensure that K needs to extend `OurComparable`.

## 13 Exceptions, Iterators, Iterables

**Exceptions**

Most likely you have encountered an exception in your code such as a `NullPointerException` or an `IndexOutOfBoundsException`. Now we will learn about how we can “throw” exceptions ourselves, and also handle thrown exceptions. Here is an example of an exception that we throw:

```
throw new RuntimeException("For no reason.");
```

*Note: Try/Catch is out of scope for now!*

Throwing exceptions is useful to notify your user of something wrong they have done. On the other hand, we can also “catch” exceptions that happen in our code! Here is an example:

```
try {
    dog.run()
} catch (Exception e) {
    System.out.println("Tried to run: " + e);
}
System.out.println("Hello World!");
```

There are a few key things to note. Firstly, the entirety of the `try` section is run until/if there is an exception thrown. If there never is an exception, the entire catch block is skipped. If there is an  exception, the code immediately jumps into the catch block with the corresponding exception, and executes from there.

**Iterators and Iterables**

These two words are very closely related, but have two different meanings that are often easy to confuse. The first thing to know is that these are both Java interfaces, with different methods that need to be implemented. Here is a simplified interface for Iterator:

```
public interface Iterator<T> {
  boolean hasNext();
  T next();
}
```

Here is a simplified interface for Iterable:

```
public interface Iterable<T> {
    Iterator<T> iterator();
}
```

Notice that in order for an object (for example an ArrayList or LinkedList) to be ***iterable***, it must include a method that returns an ***iterator***. The iterator is the object that iterates over an iterable object. Keep this relationship and distinction in mind as you work with these two interfaces.

**toString**

The `toString()` method returns a string representation of objects.

**== vs .equals**

We have two concepts of equality in Java- “==” and the “.equals()”  method. The key difference is that when using ==, we are checking if two objects have the same address in memory (that they point to the same  object). On the other hand, .equals() is a method that can be overridden by a class and can be used to define some custom way of determining  equality.

For example, say we wanted to check if two stones are equal:

```
public class Stone{
  public Stone(int weight){...}
}
Stone s = new Stone(100);
Stone r = new Stone(100);
```

If we want to consider s and r equal because they have the same  weight. If we do check equality using ==, these Stones would not be  considered equal because they do not have the same memory address.

On the other hand, if you override the equals method of Stone as follows

```
public boolean equals(Object o){
  return this.weight == ((Stone) o).weight
}
```

We would have that the stones would be considered equal because they have the same weight.

## 14 Asymptotics

**Runtime Minimization.** One of the most important properties of a program is the time it takes to execute. One goal as a programmer is to minimize the time (in seconds) that a program takes to complete.

**Runtime Measurement.** Some natural techniques:

- Measure the number of seconds that a program takes to complete using a stopwatch (either physical or in software). This tells you the actual runtime, but is dependent on the machine and inputs.
- Count the number of operations needed for inputs of a given size. This is a machine independent analysis, but still depends on the input, and also doesn’t actually tell you how long the code takes to run.
- Derive an algebraic expression relating the number of operations to the size of an input. This tells you how the algorithm scales, but does not tell you how long the code takes to run.

**Algorithm Scaling.** While we ultimately care about the runtime of an algorithm in seconds, we’ll often say that one algorithm is better than another simply because of how it scales. By scaling, we mean how the runtime of a piece of code grows as a function of its input size. For example, inserting at the beginning of ArrayList on an old computer might take $R(N) = 0.0001N$ seconds, where $N$ is the size of the list.

For example, if the runtime of two algorithms is $R_1(N) = N^2,$ and $R_2(N) = 5000 + N$, we’d say algorithm 2 is better, even though R1 is much faster for small N.

A rough justification for this argument is that performance critical situations are exactly those for which N is “large”, though this is not an obvious fact. In almost all cases we’d prefer the linear algorithm. In some limited real-world situations like matrix multiplication, one might select one algorithm for small N, and another algorithm for large N. We won’t do this in 61B.

**Simplifying Algebraic Runtime.** We utilize four simplifications to make runtime analysis simpler.

- Pick an arbitrary option to be our *cost model*, e.g. # of array accesses.
- Focus on the worst case, i.e. if the number of operations is between 1 and 2N + 1, consider only the 2N + 1.
- Ignore small inputs, e.g. treat 2N+1 just like 2N.
- Ignore constant scaling factor, e.g. treat 2N just like N.

As an example, if we have an algorithm that performs between N and 2N + 1 increment operations and between N and $4N^2 + 2N + 6$ compares, our intuitive simplifications will lead us to say that this algorithm has a runtime proportional to $N^2$.

The cost model is simply an operation that we’re picking to represent the entire piece of code. Make sure to pick an appropriate cost model!  If we had chosen the number of increment operations as our cost model, we’d mistakenly determine that the runtime was proportional to N. This is incorrect since for large N, the comparisons will vastly outnumber the increments.

**Order of Growth.** The result of applying our last 3 simplifications gives us the order of growth of a function. So for example, suppose $R(N) = 4N^2 + 3N + 6$, we’d say that the order of growth of $R(N)$ is $N^2$.

The terms “constant”, “linear”, and “quadratic” are often used for algorithms with order of growth 1, N, and $N^2$, respectively. For example, we might say that an algorithm with runtime $4N^2 + 3N + 6$ is quadratic.

**Simplified Analysis.** We can apply our simplifications in advance. Rather than computing the number of operations for ALL operations, we can pick a specific operation as our cost model and count only that operation.

Once we’ve chosen a cost model, we can either:

- Compute the exact expression that counts the number of operations.
- Use intuition and inspection to find the order of growth of the number of operations.

This latter approach is generally preferable, but requires a lot of practice. One common intuitive/inspection-based approach is use geometric intuition. For example, if we have nested for loops where i goes from 0 to N, and j goes from i + 1 to N, we observe that the runtime is effectively given by a right triangle of side length N. Since the area of a such a triangle grows quadratically, the order of growth of the runtime is quadratic.

**Big Theta.** To formalize our intuitive simplifications, we introduce Big-Theta notation. We say that a function $R(N) \in \Theta(f(N))$ if there exists positive constants $k_1$ and $k_2$ such that $k_1 f_1(N) \leq R(N) \leq k_2f_2(N)$.

Many authors write $R(N) = \Theta(f(N))$ instead of $R(N) \in \Theta(f(N))$. You may use either notation as you please. I will use them interchangeably.

An alternate non-standard definition is that $R(N) \in \Theta(f(N))$ iff the $\lim_{N\to\infty} \frac{R(N)}{f(N)} = k$, where $k$ is some positive constant.  We will not use this calculus based definition in class. I haven’t thought carefully about this alternate definition, so it might be slightly incorrect due to some calculus subtleties.

When using $\Theta$ to capture a function’s asymptotic scaling, we avoid unnecessary terms in our $\Theta$ expression. For example, while $4N^2 + 3N + 6 \in \Theta(4N^2 + 3N)$, we will usually make the simpler claim that is $4N^2 + 3N + 6 \in \Theta(N^2)$.

Big Theta is exactly equivalent to order of growth. That is, if a function $R(N)$ has order of growth $N^2$, then we also have that $R(N) \in \Theta(f(N))$.

**Runtime Analysis.** Understanding the runtime of code involves deep thought. It amounts to asking: “How long does it take to do stuff?”, where stuff can be any conceivable computational process whatsoever. It simply cannot be done mechanically, at least for non-trivial problems. As an example, a pair of nested for loops does NOT mean $\Theta(N^2)$ runtime as we saw in lecture.

**Cost Model.** As an anchor for your thinking, recall the idea of a “cost model” from last lecture. Pick an operation and count them. You want the one whose count has the highest order of growth as a function of the input size.

**Important Sums.** This is not a math class so we’ll be a bit sloppy, but the two key sums that should know are that:

- $1 + 2 + 3 + … + N \in \Theta(N^2)$
- $1 + 2 + 4 + 8 + … + N \in \Theta(N)$

**Practice.** The only way to learn this is through plenty of practice. Naturally, project 2 is going on right now, so you probably don’t have the spare capacity to be thinking too deeply, but make sure to work through the problems in lecture and below once you have room to breathe again.

## 15 Disjoint Sets

**Algorthm Development.** Developing a good algorithm is an iterative process. We create a model of the problem, develop an algorithm, and revise the performance of the algorithm until it meets our needs. This lecture serves as an example of this process.

**The Dynamic Connectivity Problem.** The ultimate goal of this lecture was to develop a data type that support the following operations on a fixed number *N* of objects:

- `connect(int p, int q)` (called `union` in our optional textbook)
- `isConnected(int p, int q)` (called `connected` in our optional textbook)

We do not care about finding the actual path between `p` and `q`. We care only about their connectedness. A third operation we can support is very closely related to `connected()`:

- `find(int p)`: The `find()` method is defined so that `find(p) == find(q)` iff `connected(p, q)`. We did not use this in class, but it’s in our textbook.

**Key observation: Connectedness is an equivalence relation.** Saying that two objects are connected is the same as saying they are in an equivalence class. This is just fancy math talk for saying “every object is in exactly one bucket, and we want to know if two objects are in the same bucket”. When you connect two objects, you’re basically just pouring everything from one bucket into another.

**Quick find.** This is the most natural solution, where each object is given an explicit number. Uses an array `id[]` of length N, where `id[i]` is the bucket number of object `i` (which is returned by `find(i)`). To connect two objects `p` and `q`, we set every object in `p`’s bucket to have `q`’s number.

- `connect`: May require many changes to `id`. Takes $\Theta(N)$ time, as algorithm must iterate over the entire array.
- `isConnected` (and `find`): take constant time.

Performing M operations takes $\Theta(MN)$ time in the worst case. If M is proportional to N, this results in a $\Theta(N^2)$ runtime.

**Quick union.** An alternate approach is to change the meaning of our `id` array. In this strategy, `id[i]` is the parent object of object `i`. An object can be its own parent. The `find()` method climbs the ladder of parents until it reaches the root (an object whose parent is itself). To connect `p` and `q`, we set the root of `p` to point to the root of `q`.

- `connect`: Requires only one change to `id[]`, but also requires root finding (worst case **$\Theta(N)$** time).
- `isConnected` (and `find`): Requires root finding (worst case $\Theta(N)$ time).

Performing M operations takes $\Theta(NM)$ time in the worst case. Again, this results in quadratic behavior if M is proportional to N.

**Weighted quick union.** Rather than `connect(p, q)` making the root of `p` point to the root of `q`, we instead make the root of the smaller tree point to the root of the larger one. The tree’s *size* is the *number* of nodes, not the height of the tree. Results in tree heights of $\lg N$.

- `connect`: Requires only one change to `id`, but also requires root finding (worst case $\lg N$ time).
- `isConnected` (and `find`): Requires root finding (worst case $\lg N$ time).

Warning: if the two trees have the same size, the book code has the opposite convention as quick union and sets the root of the second tree to point to the root of the first tree. This isn’t terribly important (you won’t be tested on trivial details like these).

**Weighted quick union with path compression.** When `find` is called, every node along the way is made to point at the root. Results in nearly flat trees. Making M calls to union and find with N objects results in no more than $O(M \log^*N)$ array accesses, not counting the creation of the arrays. For any reasonable values of N in this universe that we inhabit, $log^*(N)$ is at most 5. It is possible to derive an even tighter bound, mentioned briefly in class (known as the [Ackerman function](https://en.wikipedia.org/wiki/Ackermann_function)).

**Example Implementations**

You are not responsible for knowing the details of these implementations for exams, but these may help in your understanding of the concepts.

[QuickFind](http://algs4.cs.princeton.edu/15uf/QuickFindUF.java.html)

[QuickUnion](http://algs4.cs.princeton.edu/15uf/QuickUnionUF.java.html)

[WeightedQuickUnion](http://algs4.cs.princeton.edu/15uf/WeightedQuickUnionUF.java.html)

[Weighted Quick Union with Path Compression](http://algs4.cs.princeton.edu/15uf/QuickUnionPathCompressionUF.java.html)

## 16 BST

**Abstract Data Type.** An abstract data type (ADT) is  similar to an interface in that it is defined by its operations rather  than its implementation. It is a layer of abstraction not tied to a  particular language. Some examples of ADT’s you may have seen in class  so far include Lists, Sets, and Maps. Notice that a List can be  implemented in several different ways (i.e. LinkedList, ArrayList) and  the idea of a List is not restricted to just Java.

**Trees.** A tree consists of a set of nodes and a set of edges that connect these nodes. As there exists only one path between any two nodes, there are no cycles in a tree. If a tree is rooted,  every node except the root has exactly one parent. The root has no  parents, and a node with no children is considered a leaf.

**Binary Search Trees.** A Binary Search Tree (BST) is a rooted binary tree that maintains several key conditions to help  optimize search. For a node X, every key in the left subtree is less  than X’s key and every key in the right subtree is greater than X’s key. This aids with operations such as search since when we look for the  position of a key, we can move left or right within our tree starting  from the root depending on how our key compares to the key of each node.

**Runtime.** BST’s help optimize our search so we do not always have to look at every single element in our tree when searching  for a particular key. But how much does this optimize things? For a BST  that is “bushy” (short and fat), we can search in $O(log N)$ time where N  is the number of nodes. For a BST that is “spindly” (tall and skinny),  our search will take $O(N)$ time. This is because search time depends on  the height of our tree, where a bushy tree has a height of $log N$ and a spindly tree has a height of $N$.

## 17 B-Trees

**BSTs**

**Depth** We define the depth of a *node* as how far it is from the root. For consistency, we say the root has a depth of 0.

**Height** We define the height of a tree as the depth of the deepest node.

Notice that depending on how we insert into our BST, our height could vary drastically. We say a tree is “spindly” if it has height close to N and a tree is “bushy” if its height is closer to logN. For operations such as getting a node, we want to have the height to be as small as possible, thus favoring “bushy” BSTs.

**B-Trees**

Two specific B-Trees in this course are 2-3 Trees (A B-Tree where each node has 2 or 3 children), and 2-3-4/2-4 Trees (A B-Tree where each node has 2, 3, or 4 children). The key idea of a B-Tree is to over stuff the nodes at the bottom to prevent increasing the height of the tree. This allows us to ensure a max height of $logN$.

Make sure you know how to insert into a B-Tree. Refer back to lecture slides for examples.

With our restriction on height, we get that the runtime for contains and add are both $\Theta(logN)$

## 18 Red Black Trees

**Tree rotaions** We rotateLeft or rotateRight on a  node, creating a different but valid BST with the same elements. Notice  when we rotateLeft(G) we move the node G to be the left child of the new root.

**Left Leaning Red Black Tree** This is simply an  implementation of a 2-3 Tree with the same ideas. Be able to convert  between a 2-3 Tree and a LLRB tree. We use *red* links to  indicate two nodes that would be in the same 2-3 Node. In a left leaning RB tree, we arbitrarily enforce that edges are always to the left (for  convenience).

There are two important properties for LLRBs:

1. No node ever has 2 red links (It wouldn’t be a valid node in a 2-3 Tree if it did)
2. Every path from the root to a leaf has the same number of *black links*. This is because every leaf in a 2-3 tree has same numbers of links from root. Therefore, the tree is balanced.

**LLRB operations** Always insert with a red link at the correct location. Then use the  following three operations to “fix” or LLRB tree. See slides for visual

1. If there is a right leaning red link, rotate that node left.
2. If there are two consecutive left leaning links, rotate right on the top node.
3. If there is a node with two red links to children, flip all links with that node.

## 19 Hashing

**Brute force approach.** All data is just a sequence of bits. Can treat key as a gigantic number and use it as an array index. Requires exponentially large amounts of memory.

**Hashing.** Instead of using the entire key, represent entire key by a smaller value. In Java, we hash objects with a hashCode() method that returns an integer (32 bit) representation of the object.

**hashCode() to index conversion.** To use hashCode() results as an index, we must convert the hashCode() to a valid index. Modulus does not work since hashCode may be negative. Taking the absolute value then the modulus also doesn’t work since Math.abs(Integer.MIN_VALUE) is negative. Typical approach: use hashCode & 0x7FFFFFFF instead before taking the modulus.

**Hash function.** Converts a key to a value between 0  and M-1. In Java, this means calling hashCode(), setting the sign bit to 0, then taking the modulus.

**Designing good hash functions.** Requires a blending of sophisticated mathematics and clever engineering; beyond the scope of this course. Most important guideline is to use all the bits in the key. If hashCode() is known and easy to invert, adversary can design a sequence of inputs that result in everything being placed in one bin. Or if hashCode() is just plain bad, same thing can happen.

**Uniform hashing assumption.** For our analyses below, we assumed that our hash function distributes all input data evenly across bins. This is a strong assumption and never exactly satisfied in practice.

**Collision resolution.** Two philosophies for resolving collisions discussed in class: Separate (a.k.a. external) chaining and ‘open addressing’.

**Separate-chaining hash table.** Key-value pairs are stored in a linked list of nodes of length M. Hash function tells us which of these linked lists to use. Get and insert both require potentially scanning through entire list.

**Resizing separate chaining hash tables.** Understand how resizing may lead to objects moving from one linked list to another. Primary goal is so that M is always proportional to N, i.e. maintaining a load factor bounded above by some constant.

**Performance of separate-chaining hash tables.** Cost of a given get, insert, or delete is given by number of entries in the linked list that must be examined.

- The expected amortized search and insert time (assuming items are distributed evenly) is N / M, which is no larger than some constant (due to resizing).

**Linear-probing hash tables.** We didn’t go over this in detail in 61B, but it’s where you use empty array entries to handle collisions, e.g. linear probing. Not required for exam.

**Properties of HashCodes.** Hash codes have three necessary properties, which means  a hash code must have these properties in order to be **valid**:

1. It must be an Integer
2. If we run `.hashCode()` on an object twice, it should return the **same** number
3. Two objects that are considered `.equal()` must have the same hash code.

Not all hash codes are created equal, however. If you want your hash code to be considered a **good** hash code, it should:

1. Distribute items evenly

## 20 Heaps and Priority Queues

**Priority Queue.** A Max Priority Queue (or PQ for short) is an ADT that supports at least the insert and delete-max operations. A MinPQ supposert insert and delete-min.

**Heaps.** A max (min) heap is an array representation of a binary tree such that every node is larger (smaller) than all of its children. This definition naturally applies recursively, i.e. a heap of height 5 is composed of two heaps of height 4 plus a parent.

**Tree Representations.** Know that there are many ways to represent a tree, and that we use Approach 3b (see lecture slides) for representing heaps, since we know they are complete.

**Running times of various PQ implementations.** Know the running time of the three primary PQ operations for an unordered array, ordered array, and heap implementation.

## 21 Trees and Graph Traversals

**Trees.** A tree consists of a set of nodes and a set of edges connecting the nodes, where there is only one path between any  two nodes. A tree is thus a graph with no cycles and all vertices connected.

**Traversals.** When we iterate over a tree, we call this a “tree traversal”.

**Level Order Traversal.** A level-order traversal visits every item at level 0, then level 1, then level 2, and so forth.

**Depth First Traversals.** We have three depth first traversals: Pre-order, in-order and post-order. In a pre-order traversal, we visit a node, then traverse its children. In an in-order traversal, we traverse the left child, visit a node, then traverse the right child. In a post-order traversal, we traverse both children before visiting. These are very natural to implement recursively. Pre-order and post-order generalize naturally to trees with arbtirary numbers of children. In-order only makes sense for binary trees.

**Graphs.** A graph consists of a set of nodes and a set of edges connecting the nodes. However, unlike our tree definition, we  can have more than one path between nodes. Note that all trees are  graphs. In CS 61B, we can assume all graphs are simple graphs (AKA no  loops or parallel edges).

**Depth First Traversals.** DFS for graphs is similar to DFS for trees, but since there are potential cycles within our graph,  we add the constraint that each vertex should be visited at most once.  This can be accomplished by marking nodes as visited and only visiting a node if it had not been marked as visited already.

**Graph Traversals Overview.** Just as we had both depth-first (preorder, inorder, and postorder) traversals and a breath-first (level order) traversal for trees, we can generalize these concepts to graphs. Specifically, given a source vertex, we can “visit” vertices in:

- DFS Preorder: order in which DFS is called on each vertex.
- DFS Postorder: order in which we return from DFS calls.
- BFS: order of distance from the source. The lecture originally called this “level order” before we banish that term since nobody uses it in the real world for general graphs.

We use the term “depth first”, because we will explore “deeply” first, and use the term “breadth first” because we go wide before we go deep.

If we use BFS on a vertex of a graph that happens to be the root of a tree, we get exactly the same thing as level order traversal.

**Breadth First Search.** Unlike DFS, BFS lends itself more naturally to an iterative solution than a recursive one. When we perform BFS, we visit a source vertex s, then visit every vertex that is one link away from s, then visite very vertex that is two links away from s, and so forth.

To achieve this, we use a simple idea: Create a so-called “fringe” of vertices that we think of as the next vertices to be explored. In the case of BFS, this fringe is a Queue, since we want to visit vertices in the order that we observe them. The pseudocode is as follows:

```pseudocode
bfs(s):
    fringe.enqueue(s)
    mark(s)
    while fringe is not empty:
        dequeue(s)
        visit(s)
        for each unmarked neighbor of s:
            mark(s)
            enqueue(s)
```

In class, we discussed how we could use BFS to solve the shortest paths problem: Given a source vertex, find the shortest path from that source to every other vertex. When solving shortest paths, we add additional logic to our BFS traversal, where we also set the edgeTo for every vertex at the same time that it is marked and enqueued.

**Graph API.** In lecture, we used the Graph API from  the Princeton algorithms book. Choice of API determines how clients need to think to write codes, since certain API’s can make certain tasks  easier or harder. This can also affect runtime and memory.

**Graph Implementations.** Several graph API  implementations we explored included an adjacency matrix, list of edges, and adjacency lists. With an adjacency matrix, we essentially have a 2D array with a boolean indicating whether two vertices are adjacent. A  list of edges is simply that – a collection of all edges, such as  HashSet. The most common approach, adjacency lists, maintains an  array of lists indexed by vertex number which stores the vertex numbers  of all vertices adjacent to the given vertex.

## 22 Shortest Paths

**Dijktra’s Algorithm and Single-Source Shortest Paths.** Suppose we want to record the shortest paths from some source to every single other vertex (so that we can rapidly found a route from s to X, from s to Y, and so forth). We already know how to do this if we’re only counting the number of edges, we just use BFS.

But if edges have weights (representing, for example road lengths), we have to do something else. It turns out that even considering edge weights, we can preprocess the shortest route from the source to every vertex very efficiently. We store the answer as a “shortest paths tree”. Typically, a shortest paths tree is stored as an array of edgeTo[] values (and optionally distTo[] values if we want a constant time distTo() operation).

To find the SPT, we can use Dijkstra’s algorithm, which is quite simple once you understand it. Essentially, we visit each vertex in order of its distance from the source, where each visit consists of relaxing every edge. Informally, relaxing an edge means using it if its better than the best known distance to the target vertex, otherwise ignoring it. Or in pseudocode:

```
Dijkstra(G, s):
    while not every vertex has been visited:
        visit(unmarked vertex v for which distTo(v) is minimized)
```

Where visit is given by the following pseudocode:

```
visit(v):
    mark(v)
    for each edge e of s:
        relax(e)
```

And finally, relax is given by:

```
relax(e):
    v = e.source
    w = e.target        
    currentBestKnownWeight = distTo(w)
    possiblyBetterWeight = distTo(v) + e.weight
    if possiblyBetterWeight < currentBestKnownWeight
        Use e instead of whatever we were using before
```

Runtime is $O(V \times \log V + V \times \log V + E \times \log V)$, and since $E \gt V$ for any graph we’d run Dijkstra’s algorithm on, this can be written as more simply $O(E log V)$.  See slides for runtime description.

**A\* Single-Target Shortest Paths.** If we need only the path to a single target, then Dijkstra’s is inefficient as it explores many many edges that we don’t care about (e.g. when routing from Denver to NYC, we’d explore everything within more than a thousand miles in all directions before reaching NYC).

To fix this, we make a very minor change to Dijkstra’s, where instead of visiting vertices in order of distance from the source, we visit them in order of distance from the source + h(v), where h(v) is some heuristic.

Or in pseudocode:

```
A*(G, s):
    while not every vertex has been visited:
        visit(unmarked vertex v for which distTo(v) + h(v) is minimized)
```

It turns out (but we did not prove), that as long as h(v) is less than the true distance from s to v, then the result of A* will always be correct.

Note: In the version in class, we did not use an explicit ‘mark’. Instead, we tossed everything in the PQ, and we effectively considered a vertex marked if it had been removed from the PQ.

## 23 Minimum Spanning Trees

**Minimum Spanning Trees.** Given an undirected graph, a spanning tree T is a subgraph of G, where T is connected, acyclic, includes all vertices. The minimum spanning tree is the spanning tree whose edge weights have the smallest sum. MSTs are similar to SPTs, but despite intuition suggesting it may be the case, for many graphs, the MST is not the SPT for any particular vertex. 

**Cut Property.** If you divide the vertices up into two sets S and T (arbitrarily), then a crossing edge is any edge which has one vertex in S and one in T. Neat fact (the cut property): The minimum crossing edge for ANY cut is part of the MST.

**Prim’s Algorithm.** One approach for finding the MST is as follows: Starting from any arbitrary source, repeatedly add the shortest edge that connects some vertex in the tree to some vertex outside the tree. If the MST is unique, then the result is independent of the source (doesn’t matter where we start). 

Yet another way of thinking about Prim’s algorithm is that it is basically just Dijkstra’s algorithm, but where we consider vertices in order of the distance from the entire tree, rather than from source. Or in pseudocode, we simply change relax so that it reads:

```
relax(e):
    v = e.source
    w = e.target        
    currentBestKnownWeight = distTo(w)
    possiblyBetterWeight = e.weight // Only difference!
    if possiblyBetterWeight > currentBestKnownWeight
        Use e instead of whatever we were using before
```

Notice the difference is very subtle! Like Dijkstra’s, the runtime is $O(E log V)$. We can prove that Prim’s works because of the cut property.

**Kruskal’s Algorithm.** As an alternate algorithm and as a showcasing of various data structures in the course, we also considered Kruskal’s algorithm for finding an MST. It performs the exact same task as Prim’s, namely finding an MST, albeit in a different manner. In pseudocode, Kruskal’s algorithm is simply:

```
Initialize the MST to be empty
Consider each edge e in INCREASING order of weight:
    If adding e to the MST does not result in a cycle, add it to e
```

That’s it! The runtime for Kruskal’s, assuming that we already have all of our edges in a sorted list and use a weighted quick union with path compression to detect cycles, is $O(E \log^*V)$, or $(E \log E)$ if we have use a PQ instead. See slides for more details. We can prove that Kruskal’s works because of the cut property.

Completely unimportant technical note: We can actually make an even tighter bound than $O(E \log^*V)$ if we use the inverse Ackermann bound for WQUPC.

## 24 Multi-Dimensional Data

**Additional Set Operations** There are many other operations we might be interested in supporting on a set. For example, we might have a `select(int i)` method that returns the ith smallest item in the set. Or we might have a `subSet(T from, T to)` operation that returns all items in the set between `from` and `to`. Or if we have some notion of distance, we might have a `nearest(T x)` method that finds the closest item in the set to x.

On 1D data, it is relatively straightforward to support such  operations efficiently. If we use only one of the coordinates (e.g. X or Y coordinate), the structure of our data will fail to reflect the full  ordering of the data.

**QuadTrees** A natural approach is to make a new type of Tree– the QuadTree. The  QuadTree has 4 neighbors, Northwest,Northeast, Southwest, and Southeast. As you move your way  down the tree to support queries, it is possible to prune branches that  do not contain a useful result.

**K-D Trees** One final data structure that we have for  dealing with 2 dimensional data is the K-d Tree. Essentially the idea of a K-D tree is that it’s a normal Binary Search Tree, except we alternate what value we’re looking  at when we traverse through the tree. For example at the root everything to the left has an X value less than the root and everything to the right has a X  value greater than the root. Then on the next level, every item to the left of some  node has a Y value less than that item and everything to the right has a Y value  greater than it. Somewhat surprisingly, KdTrees are quite efficient.

## 25 Tries

**Terminology.**

- Length of string key usually represented by L.
- Alphabet size usually represented by R.

**Tries.** Analogous to LSD sort. Know how to insert and search for an item in a Trie. Know that Trie nodes typically do not contain letters, and that instead letters are stored implicitly on edge links. Know that there are many ways of storing these links, and that the fastest but most memory hungry way is with an array of size R. We call such tries R-way tries.

**Advantages of Tries.**  Tries have very fast lookup times, as we only ever look at as many characters as they are in the data we’re trying to retrieve. However, their chief advantage is the ability to efficiently support various operations not supported by other map/set implementations including:

- longestPrefixOf
- prefixMatches
- spell checking

## 26 Basic Sorts

**Inversions.** The number of pairs of elements in a sequence that are out of order. An array with no inversions is ordered.

**Selection sort.** One way to sort is by selection: Repeatedly identifying the most extreme element and moving it to the end of the unsorted section of the array. The naive implementation of such an algorithm is in place.

**Naive Heapsort.** A variant of selection sort is to use a heap based PQ to sort the items. To do this, insert all items into a MaxPQ and then remove them one by one. The first such item removed is placed at the end of the array, the next item right before the end, and so forth until that last item deleted is placed in position 0 of the array. Each insertion and deletion takes $O(\log N)$ time, and there are N insertions and deletions, resulting in a $O(N \log N)$ runtime. With some more work, we can show that heapsort is $\Theta(N \log N)$ in the worst case. This naive version of heapsort uses $\Theta(N)$ for the PQ. Creation of the MaxPQ requires $O(N)$ memory. It is also possible to use a MinPQ instead.

**In place heapsort.** When sorting an array, we can avoid the $\Theta(N)$ memory cost by treating the array itself as a heap. To do this, we first heapify the array using bottom-up heap construction (taking $\Theta(N)$ time). We then repeatedly delete the max item, swapping it with the last item in the heap. Over time, the heap shrinks from N items to 0 items, and the sorted list from 0 items to N items. The resulting version is also $\Theta(N \log N)$.

Heap construction.

$n=2^{h+1} - 1$，$h=\log_2(n+1) - 1$

$S_n=h+2^{1} \cdot(h-1)+2^{2} \cdot(h-2)+\ldots \ldots+2^{h-2} \cdot 2+2^{h-1}$

$2 S_n=2^{1} \cdot h+2^{2} \cdot(h-1)+2^{3} \cdot(h-2)+\ldots \ldots+2^{n-1} \cdot 2+2^{n}$

$S_n=-h+2^{1}+2^{2}+2^{3}+\ldots . .+2^{h-1}+2^{h}$

$S_n=-h+\left(\frac{2 \times\left(1-2^{h}\right)}{(1-2)}\right)$

$S_n=2^{{h}+1}-({h}+2)$

$S_n=(n+1)-\left(\log _{2}(n+1)-1+2\right)$

$S_n=n-\log _{2}(n+1)$

$S_n = O(n)$

**Mergesort.** We can sort by merging, as discussed in an earlier lecture. Mergesort is $\Theta(N \log N)$ and uses $\Theta(N)$ memory.

**Insertion Sort.** For each item, insert into the output sequence in the appropriate place. Naive solution involves creation of a separate data structure. The memory efficient version of this algorithm swaps items one-by-one towards the left until they land in the right place. The invariant for this type of insertion sort is that every item to the left of position i is in sorted order, and everything to the right has not yet been examined. Every swap fixes exactly one inversion.

**Insertion Sort Runtime.** In the best case, insertion sort takes $\Theta(N)$ time. In the worst case, $\Theta(N^2)$ time. More generally, runtime is no worse than the number of inversions.

**Shell’s Sort (extra slides).** Not covered on exam. Idea is to compare items that are a distance h apart from each other, starting from large h and reducing down to h=1. The last step where h=1 ensures that the array is sorted (since h=1 is just insertion sort). The earlier steps help speed things up by making long distance moves, fixing many inversions at once. Theoretical analysis of Shell’s sort is highly technical and has surprising results.

## 27 Quicksort

**Insertion Sort Sweet Spots.** We concluded our discussion of insertion sort by observing that insertion sort is very fast for arrays that are almost sorted, i.e. that have $\Theta (N)$ inversions. It is also fastest for small $N$ (roughly $N \leq 15$).

**Partitioning.** Partioning an array on a pivot means to rearrange the array such that all items to the left of the pivot are $\leq$ the pivot, and all items to the right are $\geq$ the pivot. Naturally, the pivot can move during this process.

**Partitioning Strategies.** There are many particular strategies for partitioning. You are not expected to know any particular startegy.

**Quicksort.** Partition on some pivot. Quicksort to the left of the pivot. Quicksort to the right.

**Quicksort Runtime.** Understand how to show that in the best case, Quicksort has runtime $\Theta (N \log N)$, and in the worse case has runtime $\Theta (N^2)$.

**Pivot Selection.** Choice of partitioning strategy and pivot have profound impacts on runtime. Two pivot selection strategies that we discussed: Use leftmost item and pick a random pivot. Understand how using leftmost item can lead to bad performance on real data.

**Randomization.** Accept (without proof) that Quicksort has on average $\Theta (N \log N)$ runtime. Picking a random pivot or shuffling an array before sorting (using an appropriate partitioning strategy) ensures that we’re in the average case.

**Quicksort properties.** For most real world situations, quicksort is the fastest sort.

**Hoare Partitioning.** One very fast in-place technique for partitioning is to use a pair of pointers that start at the left and right edges of the array and move towards each other. The left pointer loves small items, and hates equal or large items, where a “small” item is an item that is smaller than the pivot (and likewise for large). The right pointers loves large items, and hates equal or small items. The pointers walk until they see something they don’t like, and once both have stopped, they swap items. After swapping, they continue moving towards each other, and the process completes once they have crossed. In this way, everything left of the left pointer is $\leq$ to the pivot, and everything to the right is $\geq$ to the pivot. Finally, we swap the pivot into the appropriate location, and the partitioning is completed. Unlike our prior strategies, this partitioning strategy results in a sort which is measurably faster than mergesort.

**Selection.** A simpler problem than sorting, in selection, we try to find the Kth largest item in an array. One way to solve this problem is with sorting, but we can do better. A linear time approach was developed in 1972 called PICK, but we did not cover this approach in class, because it is not as fast as the Quick Select technique.

**Quick Select.** Using partitioning, we can solve the selection problem in expected linear time. The algorithm is to simply partition the array, and then quick select on the side of the array containing the median. Best case time is $\Theta (N)$, expected time is $\Theta (N)$, and worst case time is $\Theta (N^2)$. You should know how to show the best and worst case times. This algorithm is the fastest known algorithm for finding the median.

**Stability.** A sort is stable if the order of equal items is preserved. This is desirable, for example, if we want to sort on two different properties of our objects. Know how to show the stability or instability of an algorithm.

**Optimizing Sorts.** We can play a few tricks to speed up a sort. One is to switch to insertion sort for small problems ($\lt 15$ items). Another is to exploit existing order in the array. A sort that exploits existing order is sometimes called “adaptive”. Python and Java utilize a sort called Timsort that has a number of improvements, resulting in, for example $\Theta (N)$ performance on almost sorted arrays. A third trick, for worst case $N^2$ sorts only, is to make them switch to a worst case $N \log N$ sort if they detect that they have exceeded a reasonable number of operations.

**Shuffling.** To shuffle an array, we can assign a random floating point number to every object, and sort based on those numbers. For information on generation of random numbers, see [Fall 2014 61B](https://www.google.com/url?q=https://docs.google.com/presentation/d/1uXMsukvTUI0m5_6QfaYDmPDPXBXGRix7juEd7ekBjG0/pub?start%3Dfalse%26loop%3Dfalse%26delayms%3D3000%26slide%3Did.g46b429e30_0110&sa=D&ust=1461443429774000&usg=AFQjCNEiWI0CUmG1lyK8ZDIU6dY272cbdQ).

## 28 Sorting Bounds

**Math Problem Out of Nowhere 1.** We showed that $N! \in \Omega((N/2)^{(N/2)})$.

**Math Problem Out of Nowhere 2.** We showed that $\log(N!) \in \Omega(N \log N)$, and that $N \log N \in \Omega(\log(N!))$. Therefore $\log(N!) \in \Theta(N \log N)$.

**Seeking a Sorting Lower Bound.** We’ve found a number of sorts that complete execution in $\Theta(N \log N)$ time. This raises the obvious question: Could we do better? Let TUCS (which stands for “The Ultimate Comparison Sort”) be the best possible algorithm that compares items and puts them in order. We know that TUCS’s worst case runtime is $O(N \log N)$ because we already know several algorithm whose worst case runtime is $\Theta(N \log N)$, and TUCS’s worst case runtime is $\Omega(N)$ because we have to at least look at every item. Without further discussion, this analysis so far suggest that might be able to do better than $\Theta(N \log N)$ worst case time.

**Establishing a Sorting Lower Bound.** As a fanciful exercise, we played a game called puppy-cat-dog, in which we have to identify which of three boxes contains a puppy, cat, or dog. Since there are $3! = 6$ permutations, we need at least $ceil(\lg(6)) = 3$ questions to resolve the answer. In other words, if playing a game of 20 questions with 6 possible answers, we have to ask at least 3 questions to be sure we have the right answer. Since sorting through comparisons is one way to solve puppy-cat-dog, then any lower bound on the number of comparisons for puppy-cat-dog also applies to sorting. Given $N$ items, there are $N!$ permutations, meaning we need $\lg(N!)$ questions to win the game of puppy-cat-dog, and by extension, we need at least $\lg(N!)$ to sort N items with yes/no questions. Since $\log(N!) = \Theta(N \log N)$, we can say that the hypothetical best sorting algorithm that uses yes/no questions requires $\Omega(N \log N)$ yes/no questions. Thus, there is no comparison based algorithm that has a worst case that is a better order of growth than $\Theta(N \log N)$ compares.

## 29 Radix Sorts

**Terminology.**

- Radix - just another word for ‘base’ as in the base of a number system. For example, the radix for words written in lowercase English letters is 26. For number written in Arabic numerals it is 10.
- Radix sort - a sort that works with one character at a time (by grouping objects that have the same digit in the same position).
- Note: I will use ‘character’ and ‘digit’ interchangeably in this study guide.

**Counting Sort.**  Allows you to sort $N$ keys that are integers between $0$ and $R-1$ in $\Theta(N + R)$ time. Beats linearithmic lower bound by avoiding any binary compares. This is a completely different philosophy for how things should be sorted. This is the most important concept for this lecture.

**LSD.** In the LSD algorithm, we sort by each digit, working from right to left. Requires examination of $\Theta(WN)$ digits, where $W$ is the length of the longest key. Runtime is $\Theta(WN + WR)$, though we usually think of $R$ as a constant and just say $\Theta(WN)$. The $\Theta(WR)$ part of the runtime is due to the creation of length $R$ arrows for counting sort. We usually do LSD sort using counting sort as a subroutine, but it’s worth thinking about whether other sorts might work as well.

**LSD vs Comparison Sorting.**  Our comparison sorts, despite requiring $\Theta(N log N)$ time, can still be faster than LSD sort, which is linear in the length of our input $\Theta(WN)$. For extremely large N, LSD sort will naturally win, but $\log N$ is typically pretty small. Know which algorithm is best in the two extreme cases of very long dissimilar strings and very long, nearly equal strings.

**MSD.** In MSD sorting, we work from left to right, and solve each resulting subproblem independently. Thus, for each problem, we may have as many as R subproblem. Worst case runtime is exactly the same as LSD sort, $\Theta(WN + WR)$, though can be much better. In the very best case, where we only have to look at the top character (only possible for R > N), we have a runtime of $\Theta(N + R)$.

## 30 Sorting and Data Structures Conclusion

**Radix Sort vs. Comparison Sorts.** In lecture, we used the number of characters examined as a cost model to compare radix sort and comparison sort. For MSD Radix Sort, the worst case is that each  character is examined once for $NM$ characters examined. For merge sort,  $MN\log N$ is the worst case characters examined. Thus, we can see that  merge sort is slower by a factor of $\log N$ if character comparisons are an appropriate cost model. Using an empirical analysis, however, we saw  that this does not hold true because of lots of background reasons such  as the cache, optimized methods, extra copy operations, and overall  because our cost model does not account for everything happening.

**Just-In-Time Compiler** The “interpreter” studies your code as it runs so that when a sequence of code is run many times, it  studies and re-implements based on what it learns while running to  optimize it. For example, if a LinkedList is created many times in a  loop and left unused, it eventually learns to stop creating the  LinkedLists since they are never used. With the Just-In-Time compiler  disabled, merge sort, from the previous section, is indeed slower than  MSD Radix Sort.

**Radix Sorting Integers.**  When radix sorting  integers, we no longer have a `charAt` method. There are lots of  alternative options are stilizing mods and division to write your own  `getDigit()` method or to make each Integer into a String. However, we  don’t actually have to stick to base 10 and can instead treat the  numbers as base 16, 256, or even base 65536 numbers. Thus, we can reduce the number of digits, which can reduces the runtime since runtime for  radix sort depends on alphabet size.

**Summary.** The sort problem is to take a sequence of objects and put them into the correct order. The search problem is to store a collection of objects such that they can be rapidly retrieved (i.e. how do we implement a Map or Set). We made the obersvation that BST maps are roughly analagous to comparison based sorting, and hash maps are roughly analagous to counting based (a.k.a. integer) sorting. We observed that we have a 3rd type of sort, which involves sorting by digit, which raised the question: What sort of data structure is analogous to LSD or MSD sort? Tries.

## 31 Dynamic Programming

**DAGs.** One very special type of graph is the directed acyclic graph, also called a [DAG](https://en.wikipedia.org/wiki/Directed_acyclic_graph).

**DAGSPT.** While Dijkstra’s will work fine on any DAG with non-negative edges, and will complete in $O(E \log V + V \log V)$ time, we can do slightly better performance wise using the DAGSPT algorithm. DAGSPT simply consists of first finding a topological ordering of the vertices, then relaxing all of the edges from each vertex in topological order. The runtime of this algorithm is $\Theta(E + V)$. This algorithm does not work general graphs, only DAGs. This algorithm works, even if we have negative edges.

**Dynamic Programming.** Dynamic programming is the following process:

- Identify a collection of subproblems.
- Solve subproblems, working from smallest to largest.
- Use the answers from the smaller problems to help solve the larger ones.

**Dynamic Programming and SPT.** The DAGSPT algorithm is an example of dynamic programming. We start by solving the simplest subproblem (distance from the source to the source), and then use the results of that subproblem to solve larger subproblems (distance from the source to vertices two edges away), and so forth.

**Longest Increasing Subsequence.** As an example of Dynamic Programming, we studied the longest increasing subsequence (LIS) problem. This is not because the LIS problem is particularly important, but rather than its one of the easiest non-trivial DP problems. The LIS problem is: Given a sequence of numbers, find the longest (not necessarily contiguous) subsequence that is increasing. For example, given the sequence [6, 2, 8, 4, 5, 7], the LIS is [2, 4, 5, 7]. The length of the longest increasing subsequence (LLIS) problem says to simply find the length of the LIS. For example, given the sequence [6, 2, 8, 4, 5, 7], the LLIS is 4.

**The LLIS Problem and Longest Path.** We can represent an instance of the LLIS problem as a DAG. We create one vertex per item in the sequence numbered from 0 to N - 1. We draw an arrow from i to j if the ith element of the sequence is less than the jth element. The solution to the LLIS problem is then simply: Find the path with the most edges in this DAG, starting from any vertex.

**Solving the Most-Edges Path Problem.** One solution to the most-edges path problem is to set all edge weights to -1, then use the DAGSPT algorithm to find the shortest-paths-tree from each vertex. Since the edge weights all have weight -1, the shortest path in each SPT will be the one with the most edges. Or in other words, given the SPT for vertex i, the LLIS that starts at item i will be given by the absolute value of the minimum of the `distTo` array. In class, I called this DAG problem “longest path”, but in retrospect, I think “most edges path” is a clearer name, so I will use it throughout this study guide instead.

**Implementing Our Most-Edges Based Approach.** Our LLIS algorithm thus consists of first forming a DAG with edges of length -1, then finding a topological ordering on this graph, then running DAGSPT N times and recording the minimum of the `distTo` array, and finally returning 1 + the absolute value of the minimum of these minimums. The runtime of this algorithm is $O(N^3)$. See the B level problems and/or extra slides for why.

**Reduction.** Transforming a problem from one domain and solving it in that new domain is sometimes called “reduction”. For example, we reduced the LLIS problem to N-solutions of the most-edges on a DAG problem. Other informal examples of reduction: we can reduce “illuminating a room” to “flipping a light switch”, we can reduce “getting to work” to “riding BART”, we can reduce “8puzzle” to “A*”.

**Inefficiency of Using Most-Edges to Solve LLIS.** Reduction to N-solutions of the most-edges path problem works, but is inefficient. In particular, we observe that the latest shortest paths trees are in fact sub-trees of the earlier shortest paths problem. In effect, we are solving a bunch of subproblems in the wrong order!

**LLIS Using DP.** Our earlier approach boiled down to solving the LLIS problem *starting* from each separate vertex V. However, if we solve the LLIS problem *ending* at each vertex V, we can save ourselves lots of work. Define Q(K) to be the LLIS ending at vertex number K. For example, given the sequence [6, 2, 8, 4, 5, 7], Q(3) is 2, since the length of the longest path ending at 4 (which is item #3) is 2. Calculating these Q(K) values are the subproblems of our DP approach.

**Using Smaller Qs to Solve Larger Qs.** For our approach to be dynamic programming, we must be able to solve our larger subproblems in terms of our smaller ones. Trivially, we have that Q(0) is 1, since the LLIS ending at vertex 0 is 1. For subsequent Q(K), we have that Q(K) is equal to 1 + Q(M), where M is the vertex such that Q(M) is the largest known Q value such that there is an edge from K to M (equivalently that the Kth item is less than the Mth item).

**Using Qs to Solve Our DP.** Supposing that we have calculated Q(K) for all K. Recalling that Q(K) is defined as the length of the largest subsequence ending at K, then the length of the largest subsequence is simply the maximum of all Q(K).

**Implementing our LLIS DP Algorithm.** While the DAG is a useful abstraction to guide our thinking, it is ultimately unnecessary. Our solution to LLIS can be simply presented as:

- Create Q, an array of length N. Set Q[0] to 1, and Q[K] to negative infinity.
- For each K = 1, …, N, then for each item L = 0, … K - 1, if item L is less than item K, and if Q[L] + 1 > Q[K], then set Q[K] = Q[L] + 1

The runtime or this algorithm is simply $\Theta(N^2)$.

## 31 Compression 

**Compression Model #1: Algorithms Operating on Bits.** Given a sequence of bits B,  we put them through a compression algorithm C to form a new bitstream C(B). We can run C(B) through a corresponding decompression algorithm to recover B. Ideally,  C(B) is less than B.

**Variable Length Codewords.** Basic idea: Use variable length codewords to represent symbols, with shorter keywords going with more common symbols. For example, instead of representing every English character by a 8 bit ASCII value, we can represent more common values with shorter sequences. Morse code is an example of a system of variable length codewords.

**Prefix Free Codes.** If some codewords are prefixes of others, then we have ambiguity, as seen in Morse Code. A prefix free code is a code where no codeword is a prefix of  any other. Prefix free codes can be uniquely decoded.

**Shannon-Fano Coding.** Shannon-Fano coding is an intuitive procedure for generating a prefix free code. First, one counts the occurrence of all symbols. Then you recursively split characters into halves over and over based on  frequencies, with each half either having a 1 or a 0 appended to the end  of the codeword.

**Huffman Coding.** Huffman coding generates a provably optimal prefix free code,  unlike Shannon-Fano, which can be suboptimal. First, one counts the occurrence of all symbols, and create a “node” for each symbol. We then merge the two lowest occurrence nodes into a tree with a new supernode as root, with each half either having a 1 or a 0 appended to the beginning of the codeword. We repeat this until all symbols are part of the tree. Resulting code is optimal.

**Huffman Implementation.** To compress a sequence of symbols, we count frequencies, build an encoding array and a decoding trie, write the trie to the output, and then look up each symbol in the encoding array and write out the appropriate bit sequence to the output. To decompress, we read in the trie, then repeatedly use longest prefix matching to recover the original symbol.

**General Principles Behind Compression.** Huffman coding is all about representing common symbols with a small number of bits. There are other ideas, like run length encoding where you replace every character by itself followed by its number of occurrences, and LZW which searches for common repeated patterns in the input. More generally, the goal is to exploit redundancy and existing order in the input.

**Universal Compression is Impossible.** It is impossible to create an algorithm that can compress any bitstream by 50%. Otherwise, you could just compress repeatedly until you ended up with just 1 bit, which is clearly absurd. A second argument is that for an input bitstream of say, size 1000, only 1 in 2^499 is capable of being compressed by 50%, due to the pigeonhole principle.

**Compression Model #2: Self Extracting Bits.** Treating the algorithm and the input  bitstream separately (like we did in model #1) is a more accurate model, but it seems to leave open strange algorithms like one in which we simply hardcode our desired output into the algorithm itself. For example, we might have a .java decompression algorithm  that has a giant `byte[]` array of your favorite TV show, and if the algorithm gets the input `010`, it outputs this `byte[]` array.

In other words, it seems to make more sense to include not just the compressed bits when considering the size of our output, but also the algorithm used to do the decompression.

One conceptual trick to make this more concrete is to imagine that our algorithm and the bits themselves are a single entity, which we can think of a self-extracting bit sequence. When fed to an interpreter, this self-extracting bit sequence generates a particular output sequence.

**Hugplant Example.** If we have an image file of something like the hugplant.bmp from lecture, we can break it into 8 bit chunks and  then Huffman encode it. If we give this file to someone else, they probably won’t know  how to decompress it, since Huffman coding is not a standard compression algorithm supported by major operating systems. Thus, we also need to provide the Huffman  decoding algorithm. We could send this as a separate .java file, but for conceptual convenience and in line with compression model #2, we’ll imagine that we have packaged our compressed bit stream into a `byte[]` array in a .java file. When passed to an interpreter, this bitstream yields the original hugplant.bmp, which is 4 times larger than the compressed bitstream + huffman interpreter.
