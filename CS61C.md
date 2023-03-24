# Great Ideas in Computer Architecture

 (a.k.a. Machine Structures)

![image-20220424152527440](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625768.png)

![image-20220424152630454](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625769.png)

1 Abstraction (Layers of Representation / Interpretation)

![image-20220425123316102](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625770.png)

2 Moore’s Law

3 Principle of Locality/Memory Hierarchy

![image-20220430154600475](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625771.png)

4 Parallelism

5 Performance Measurement and Improvement

6 Dependability via Redundancy

## Number Representations

- If result of add (or -, *, / ) **cannot be represented by these rightmost HW bits**, we say **overflow** occurred.

- We represent “things” in computers as particular bit patterns: N bits ⇒ 2^N^ things

- These 5 integer encodings have different benefits

  - One’s Complement and Sign and Magnitude have most problems

  - **Unsigned**

  - **Two’s Complement**

    - 2^N-1^ **non**-negatives
    - 2^N-1^ negatives
    - one zero
  
  - **Bias Encoding**
  
    - \# = unsigned + bias
    - Bias for N bits chosen as –(2^N-1^-1)
    - one zero
    - 2^N-1^ **non**-positives

## The C Programming Language

- Storing struct: **word alignment**

- Utilizing indirection and avoiding maintaining two copies of the number 10

  ```c
  int ARRAY_SIZE = 10; 
  int i, a[ARRAY_SIZE]; 
  for(i = 0; i < ARRAY_SIZE; i++){ ... }
  ```

- An array in C does not know its own length, & bounds not checked!
  
  - We must pass the array and its size to a procedure which is going to traverse it.
  
- pointer + n
  
- Adds `n*sizeof("whatever pointer is pointing to")` to the memory address
  
- If you want to change a value in a procedure, **pass a pointer**.
  - ![](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625772.png)
  - ![](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625773.png)

- `sizeof ` knows the size of arrays

- Dynamic Memory Allocation

  - Use `malloc()` and `free()` to allocate and deallocate memory from heap.

  - `malloc()`: Allocates raw, uninitialized memory from heap.

    - `ptr = (int *) malloc (n*sizeof(int));` This allocates an array of n integers.

  - Even though the program frees all memory on exit (or when main returns), don’t be lazy to `free(ptr)`!

  - The following two things will cause your program to crash or behave strangely later on, and cause VERY VERY hard to figure out bugs: 

    - `free()` the same piece of memory twice
    - calling `free()` on something you didn’t get back from `malloc()`
    - The runtime does not check for these mistakes

  - `realloc(p, size)`

    - If `p` is `NULL`, then `realloc` behaves like `malloc`

    - If `size` is `0`, then `realloc` behaves like `free`, deallocating the block from the heap

      ```c
      int *ip; 
      ip = (int *) malloc(10*sizeof(int)); 
      /* always check for ip == NULL */ 
      … … … 
      ip = (int *) realloc(ip,20*sizeof(int)); 
      /* always check NULL, contents of first 10 elements retained */ 
      … … … 
      realloc(ip,0); 
      /* identical to free(ip) */
      ```

- C is an efficient language, with little protection

  - Array bounds not checked
  - Variables not automatically initialized

- Memory Locations

  - What is stored?

    - **Structure declaration does not allocate memory**
    - Variable declaration does allocate memory

  - A program’s address space contains 4 regions: 

    - stack: grows downward
      - If declared outside a procedure (global), allocated in “static” storage
    - heap: space requested for pointers via malloc() ; resizes dynamically, grows upward
    - static data: does not grow or shrink
      - If declared inside procedure (local), allocated on the “stack” and freed when procedure returns.
    - code: loaded when program starts, does not change
    - ![image-20220401162945036](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625774.png)

  - Stack

    - Stack frame includes: 
      - Return “instruction” address
      - Parameters
      - Space for other local variables
    - Stack frames contiguous blocks of memory; stack pointer tells where top stack frame is
    - When procedure ends, stack frame is tossed off the stack; frees memory for future stack frames

    ![image-20220401234517366](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625775.png)

  - The Heap (Dynamic memory)
    - Large pool of memory, not allocated in contiguous order
      - back-to-back requests for heap memory could result blocks very far apart

- Pointer Errors

  - Dangling reference (use ptr before malloc) 

  - Memory leaks (tardy free, lose the ptr) 

  - Writing off the end of arrays

  - Returning Pointers into the Stack

    ![image-20220402093922527](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625776.png)

  - Use After Free

    - https://stackoverflow.com/questions/42588482/c-accessing-data-after-memory-has-been-freeed

  - Forgetting realloc Can Move Data
  
    ```c
    struct foo *f = malloc(sizeof(struct foo) * 10); 
    ...
    struct foo *g = f;
    .... 
    f = realloc(sizeof(struct foo) * 20);
    ```
    - Result is g may now point to invalid memory
  
  - Freeing the Wrong Stuff
  
  - Double-Free
  
  - Losing the initial pointer
  
    ```c
    int *plk = NULL; 
    void genPLK() { 
        plk = malloc(2 * sizeof(int)); 
        … … … 
        plk++;
        /* This MAY be a memory leak if we don't keep somewhere else a copy of the original malloc'ed pointer */
    }
    ```

- Valgrind
  - Valgrind slows down your program by an order of magnitude, but... 
  - It adds a tons of checks designed to catch most (but not all) memory errors   
    - Memory leaks   
    - Misuse of free   
    - Writing over the end of arrays
  - Tools like Valgrind are absolutely essential for debugging C code

## Floating Point    

- What can we represent in N bits?
  - Unsigned integers
    - 0 to 2^N-1^ 
    - (for N=32, 2^N-1^ = 4,294,967,295)
  - Signed Integers (Two’s Complement)
    - -2^(N-1)^ to 2^(N-1)^ - 1
    - (for N=32, 2^(N-1)^ - 1 = 2,147,483,647)
- Scientific Notation (in Binary)
  - ![image-20220404212145082](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625777.png)
- Floating Point Representation 

  - With floating point representation, each numeral carries an exponent field recording the whereabouts of its binary point.

  - The binary point can be outside the stored bits, so very large and small numbers can be represented.
  - ![image-20220404212303318](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625778.png)
- Overflow & Underflow
  - ![image-20220404212409496](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625779.png)
- [IEEE 754 Floating Point Standard](https://www.h-schmidt.net/FloatConverter/IEEE754.html)
  - Biased Notation
    - Bias is number subtracted to get real number.
      - IEEE 754 uses bias of 127 for single prec.
    - Wanted bigger (integer) exponent field to represent bigger numbers.
    - 2’s complement poses a problem (because negative numbers look bigger)
    - We’re going to see that the numbers are ordered EXACTLY as in sign-magnitude
      - i.e., counting from binary odometer 00…00 up to 11…11 goes from 0 to +MAX to -0 to -MAX to 0
  - ![image-20220404213419805](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625780.png)
- Special Numbers
  - ![image-20220406193558797](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625781.png)
  -  What do I get if I calculate sqrt(-4.0) or 0/0? If ∞ not an error, these shouldn’t be either.
    - Not a Number (NaN)
      - Hope NaNs help with debugging?
        - They contaminate: op(NaN, X) = NaN
        - Can use the significand to identify which!
  - Denorms
    - ![image-20220406193926199](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625782.png)
    - ![image-20220406194113780](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625783.png)
- Understanding the Significand
  - Method 1 (Fractions)
    - ![image-20220406195257035](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625784.png)
  - Method 2 (Place Values)
    - ![image-20220406195317272](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625785.png)

- Floating Point add is not associative!
  - ![image-20220406195831041](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625786.png)

- Precision and Accuracy
  - **Precision** is a count of the number bits in used to represent a value.
  - **Accuracy** is the difference between the actual value of a # and its computer representation.
  - High precision permits high accuracy but doesn’t guarantee it.
    - It is possible to have high precision but low accuracy.
    - Example: `float pi = 3.14;`
    - pi will be represented using all 24 bits of the significant (highly precise), but is only an approximation (not accurate).
- IEEE FP Rounding Modes
  - ![image-20220406201647488](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625787.png)

- Casting floats to ints and vice versa
  - ![image-20220406203324204](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625788.png)

- Double Precision Fl. Pt. Representation
  - ![image-20220406204212109](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625789.png)

- Other Floating Point Representations

## RISC-V Assembly Language

- Different CPUs implement different sets of instructions. The set of instructions a particular CPU implements is an Instruction Set Architecture (ISA).

- Instruction set for a particular architecture (e.g. RISC-V) is represented by the Assembly language

- Each line of assembly code represents one instruction for the computer.

- In RISC-V, all instructions are 4 bytes, and stored in memory just like data.

- Assembly Variables: Registers
  - Assembly operands are registers
  - Benefit: Since registers are directly in hardware, they’re very fast (faster than 0.25ns)
  - Drawback: Since registers are in hardware, there is a predetermined number of them
  - Each RISC-V register is 32 bits wide (in RV32 variant)
    - Groups of 32 bits called a word in RV32
  - In assembly language, the registers have no type
    - Operation determines how register contents are treated
  - ![image-20220407192611588](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625790.png)

- Assembly Instructions

  - Unlike in C (and most other high-level languages), each line of assembly code contains at most 1 instruction

  - Syntax of Instructions

    - `one two, three, four`  `add x1, x2, x3`

    - one = operation by name
    - two = operand getting result (“destination,” x1) rd
    - three = 1st operand for operation (“source1,” x2) rs1
    - four = 2nd operand for operation (“source2,” x3) rs2

  - Immediates imm

    - `addi x3,x4,10`
    - 32bit
    - There is no Subtract Immediate in RISC-V: Why?
      - Limit types of operations that can be done to absolute minimum
      - if an operation can be decomposed into a simpler operation, don’t include it
      - `addi x3,x4,-10`

  - Register Zero

    - One particular immediate, the number zero (0), appears very often in code.
    - the register zero (x0) is ‘hard-wired’ to value 0
      - `add x3,x4,x0`
    - Defined in hardware, so an instruction `add x0,x3,x4` will not do anything!

- Memory Addresses are in Bytes
  - 8 bit chunk is called a byte (1 word = 4 bytes)
  - Word address is same as address of rightmost byte – least-significant byte (i.e. Little-endian convention)
  - ![image-20220407195559912](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625791.png)

- Big Endian vs. Little Endian
  - Consider the number 1025 as we typically write it:
  - ![image-20220407195948923](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625792.png)

- Data Transfer: Load from and Store to memory

  - ![image-20220407195300794](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625793.png)

  - Speed of Registers vs. Memory

    - How much faster are registers than DRAM??
      - About 50-500 times faster! (in terms of latency of one access - tens of ns)
      - But subsequent words come every few ns

  - Load from Memory to Register

    - Using Load Word (lw)
      - `int A[100]; g = h + A[3];`
      - `lw x10,12(x15) # Reg x10 gets A[3]`
      - x15 – base register (pointer to A[0]) 
      - 12 – offset in bytes
      - Offset must be a constant known at assembly time

  - Store from Register to Memory

    - ```assembly
      lw x10,12(x15) 			# Temp reg x10 gets A[3]
      add x10,x12,x10 		# Temp reg x10 gets h + A[3]
      sw x10,40(x15) 			# A[10] = h + A[3]
      ```

  - load byte: lb store byte: sb

    - ![image-20220407201350442](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625794.png)
    - What is in x12 ?
      - ![image-20220407202142069](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625795.png)

- Decision Making

  - `beq reg1,reg2,L1` means: go to statement labeled L1 if (value in reg1) == (value in reg2)
  - `beq` stands for branch if equal
  - ` bne` for branch if not equal

- Types of Branches

  - Branch – change of control flow
  - Conditional Branch – change control flow depending on outcome of comparison
    - branch if equal (beq) or branch if not equal (bne)
    - Also branch if less than (`blt`) and branch if greater than or equal (`bge`)
    - And unsigned versions (bltu, bgeu)

  - Unconditional Branch – always branch
    - a RISC-V instruction for this: jump (j), as in j label

- Loops in C/Assembly
  - There are three types of loops in C: while, do … while, for
  - Each can be rewritten as either of the other two, so the same branching method can be applied to these loops as well.
  - Key concept: Though there are multiple ways of writing a loop in RISC-V, the key to decision-making is conditional branch
  - ![image-20220407203222097](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625796.png)

- Logical Instructions
  - Operations to pack /unpack bits into words
  - `and or xor sll srl sra srai` 
    - Shift left logical, inserting 0’s on right
    - Shift right logical
    - Shift right arithmetic, insert high-order sign bit into empty bits
      - Unfortunately, this is NOT same as dividing by 2^n^
        - Fails for odd negative numbers
        - C arithmetic semantics is that division should round towards 0
  - Register: `and x5, x6, x7 # x5 = x6 & x7`
  - Immediate: `andi x5, x6, 3 # x5 = x6 & 3`
  - No NOT in RISC-V
    - Use `xor` with `11111111two`

- Assembler to Machine Code
  
- ![image-20220407204124436](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625797.png)
  
- Program Execution
  - Instruction is fetched from memory, then control unit executes instruction using datapath and memory system, and updates PC
  - (default add +4 bytes to PC, to move to next sequential instruction; branches, jumps alter)
  - PC (program counter) is a register internal to the processor that holds byte address of next instruction to be executed
  - ![image-20220407204815636](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625798.png)

- Helpful RISC-V Assembler Features
  - Symbolic register names
    - E.g., a0-a7 for argument registers (x10-x17) for function calls
    - E.g., zero for x0
  - Pseudo-instructions
  - Shorthand syntax for common assembly idioms
    - E.g., mv rd, rs = addi rd, rs, 0
    - E.g., li rd, 13 = addi rd, x0, 13
    - E.g., nop = addi x0, x0, 0

- Six Fundamental Steps in Calling a Function

  - Put **arguments** in a place where function can access them
  - Transfer control to function
  - Acquire (local) storage resources needed for function
  - Perform desired task of the function
  - Put **return value** in a place where calling code can access it and restore any registers you used; release local storage
  - Return control to point of origin, since a function can be called from several points in a program

- Registers faster than memory, so use them

  - `a0–a7 (x10-x17)`: eight **argument** registers to pass parameters and two return values `(a0-a1)`
  - `ra`: one **return address** register to return to the point of origin `(x1)`
  - Also `s0-s1 (x8-x9)` and `s2-s11 (x18-x27)`: saved registers

- Instruction Support for Functions

  - ![image-20220408224058754](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625799.png)
    - Why use jr here? Why not use j?
      - Answer: sum might be called by many places, so we can’t return to a fixed place. The calling proc to sum must be able to say “return here” somehow.
  - Single instruction to jump and save return address: jump and link (`jal`)
    - ![image-20220408224448735](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625800.png)
  - Why have a jal? 
    - Make the common case fast: function calls very common
    - Reduce program size
    - Don’t have to know where code is in memory with jal!
  - Invoke function: jump and link instruction (jal) (really should be laj link and jump)
    - **link means form an address or link that points to calling site to allow function to return to proper address**
    - Jumps to address and simultaneously saves the address of the **following** instruction in register ra
  - `jalr rd, rs, imm` – jump-and-link register
  - Return from function: jump register instruction (jr)
    - Unconditional jump to address specified in register:` jr ra`
    - Assembler shorthand: `ret = jr ra`

- Stack

  - `sp` is the stack pointer in RISC-V (`x2`)

  - Convention is grow stack down from high to low addresses

    - Push decrements `sp`, Pop increments `sp`

  - Stack frame includes:

    - Return “instruction” address
    - Parameters (arguments)
    - Space for other local variables

  - Stack frames contiguous blocks of memory; stack pointer tells where bottom of stack frame is

  - When procedure ends, stack frame is tossed off the stack; frees memory for future stack frames

    - ![image-20220408230907410](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625801.png)

  - ```
    int Leaf (int g, int h, int i, int j) {
    	int f;
    	f = (g + h) – (i + j);
    	return f;
    }
    ```

    - ![image-20220408231046582](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625802.png)
    - Need to save old values of s0 and s1
    - ![image-20220408231122311](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625803.png)

- Register Conventions 
  - CalleR: the calling function
  - CalleE: the function being called
  - When callee returns from executing, the caller needs to know which registers may have changed and which are guaranteed to be unchanged.
    - Register Conventions: A set of generally accepted rules as to which registers will be unchanged after a procedure call (jal) and which may be changed.
  - To reduce expensive loads and stores from spilling and restoring registers, RISC-V function-calling convention divides registers into two categories:
    - Preserved across function call
      - Caller can rely on values being unchanged 
      - `sp, gp, tp, "saved registers" s0-s11 (s0 is also fp)`
    - Not preserved across function call
      - Caller cannot rely on values being unchanged
      - `Argument/return registers a0-a7, ra, "temporary registers" t0-t6`
  - RISC-V Symbolic Register Names
    - ![image-20220408231415955](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625804.png)
  
- Allocating Space on Stack
  - C has two storage classes: automatic and static
    - **Automatic** variables are local to function and discarded when function exits
    - **Static** variables exist across exits from and entries to procedures
  - Use stack for automatic (local) variables that don’t fit in registers
  - **Procedure frame** or **activation record**: segment of stack with saved registers and local variables
- Stack Before, During, After Function
  
- ![image-20220408231813728](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625805.png)
  
- Using the Stack
  - To use stack, we decrement this pointer by the amount of space we need and then fill it with info
  - ![image-20220408231946736](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625806.png)

- When a C program is run, there are three important memory areas allocated:
  - Static: Variables declared once per program, cease to exist only after execution completes - e.g., C globals
  - Heap: Variables declared dynamically via malloc
  - Stack: Space to be used by procedure during execution; this is where we can save register values
- RV32 Memory Allocation
  
- ![image-20220408232224610](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625807.png)
  
- RV32 So Far…
  
  - ![image-20220408233024978](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625808.png)

## RISC-V Instruction Representation

- Big Idea: Stored-Program Computer
  - Instructions are represented as bit patterns – can think of these as numbers
  - Therefore, entire programs can be stored in memory to be read or written just like data
  - Consequence #1: Everything Has a Memory Address
    - Since all instructions and data are stored in memory, everything has a memory address: instructions, data words
      - Both branches and jumps use these
    - C pointers are just memory addresses: they can point to anything in memory
      - Unconstrained use of addresses can lead to nasty bugs; avoiding errors up to you in C; limited in Java by language design
    - One register keeps address of instruction being executed: “Program Counter” (PC)
      - Basically a pointer to memory
      - Intel calls it Instruction Pointer (IP)
    - Consequence #2: Binary Compatibility
      - Programs are distributed in binary form
        - Programs bound to specific instruction set
        - Different version for phones and PCs
      - New machines want to run old programs (“binaries”) as well as programs compiled to new instructions
      - Leads to “backward-compatible” instruction set evolving over time
- Instructions as Numbers
  - Most data we work with is in words (32-bit chunks):
    - Each register is a word
    - lw and sw both access memory one word at a time
  - So how do we represent instructions?
  - RISC-V seeks simplicity: since data is in words, make instructions be fixed-size 32-bit words also
    - Same 32-bit instructions used for RV32, RV64,RV128
  - One word is 32 bits, so divide instruction word into “fields”
  - Each field tells processor something about instruction
  - We could define different fields for each instruction, but RISC-V seeks simplicity, so define six basic types of instruction formats:
    - R-format for register-register arithmetic operations
    - I-format for register-immediate arithmetic operations and loads
    - S-format for stores
    - B-format for branches (minor variant of S-format)
    - U-format for 20-bit upper immediate instructions
    - J-format for jumps (minor variant of U-format)
  - Summary of RISC-V Instruction Formats
    - ![image-20220420170115234](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625809.png)

- Complete RV32I ISA!
  - ![image-20220420170247646](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625810.png)

## Compiling, Assembling, Linking, and Loading

- Representation/Interpretation
  - Interpreter is a program that executes other programs
  - ![image-20220421120654955](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625811.png)
  
  - interpret a high-level language when efficiency is not critical
  - translate to a lower-level language to increase performance
  - How do we run a program written in a source language?
    - Interpreter: Directly executes a program in the source language
      - Python interpreter is just a program that reads a python program and performs the functions of that python program
      - Interpreter slower (10x?), code smaller (2x?)
      - Interpreter closer to high-level, so can give better error messages
      - Interpreter provides instruction set independence: run on any machine
    - Translator: Converts a program from the source language to an equivalent program in another language
      - Translated/compiled code almost always more efficient and therefore higher
        performance: Important for many applications, particularly operating systems
      - Translation/compilation helps “hide” the program “source” from the users

- Compiler
  - Input: High-Level Language Code (e.g., foo.c)
  - Output: Assembly Language Code (e.g., foo.s for RISC-V)
  - Note: Output may contain pseudo-instructions

- Assembler
  - Input: Assembly Language Code (includes pseudo ops) (e.g., foo.s for RISC-V)
  - Output: Object Code, information tables (true assembly only) (e.g., foo.o for RISC-V)
  - Reads and Uses Directives
  - Replace Pseudo-instructions
  - Produce Machine Language
    - Branch instructions can refer to labels that are “forward” in the program
      - Solved by taking two passes over the program
        - First pass remembers position of labels
        - Second pass uses label positions to generate code
    - Symbol Table
      - List of “items” in this file that may be used by other files
      - Labels: function calling
      - Data: anything in the .data section; variables which may be accessed across files
    - Relocation Table
      - List of “items” whose address this file needs
      - Any absolute label jumped to: jal, jalr
      - Any piece of data in static section
  - Creates Object File
    - A standard format is ELF
      - object file header: size and position of the other pieces of the object file
      - text segment: the machine code
      - data segment: binary representation of the static data in the source file
      - relocation information: identifies lines of code that need to be fixed up later
      - symbol table: list of this file’s labels and static data that can be referenced
      - debugging information

- Linker
  - Input: Object code files, information tables (e.g., foo.o,libc.o for RISC-V)
  - Output: Executable code (e.g., a.out for RISC-V)
  - Combines several object (.o) files into a single executable (“linking”)
  - Enable separate compilation of files
  - Step 1: Take text segment from each .o file and put them together
  - Step 2: Take data segment from each .o file, put them together, and concatenate this onto end of text segments
  - Step 3: Resolve references
    - Go through Relocation Table; handle each entry
    - I.e., fill in all absolute addresses
  - Four Types of Addresses
    - ![image-20220421123010019](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625812.png)
  - Output of linker: executable file containing text and data (plus header)

- Dynamically Linked Libraries

  - ![image-20220421123146047](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625813.png)

  - ![image-20220421123203007](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625814.png)

  - ![image-20220421123235240](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625815.png)

- Loader
  - Input: Executable Code (e.g., a.out for RISC-V)
  - Output: (program is run)
  - Executable files are stored on disk
  - When one is run, loader’s job is to load it into memory and start it running
  - In reality, loader is the operating system (OS)
  - Loading is one of the OS tasks
  - ![image-20220421123403771](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625816.png)

- ![image-20220421124248203](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625817.png)

## Introduction to Synchronous Digital Systems (SDS)

- Machine Structures

  - ![image-20220421154403593](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625818.png)
- New-School Machine Structures

  - ![image-20220421154505759](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625819.png)
  - ![image-20220421154535704](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625820.png)
- Synchronous Digital Systems

  - Synchronous: 
    - All operations coordinated by a central clock
    - “Heartbeat” of the system!
    - Clocks control pulse of our circuits
  - Digital: 
    - All values represented by discrete values 
    - Electrical signals are treated as 1s and 0s; grouped together to form words
    - Voltages are analog, quantized to 0/1
- Switches: Basic Element of Physical Circuit
- Transistors

  - Modern digital systems designed in CMOS
    - MOS: Metal-Oxide on Semiconductor
    - C for complementary: normally-open and normally-closed switches
  - MOS transistors act as voltage-controlled switches
- Signals and Waveforms

  - Signals
    - When **digital** is only treated as 1 or 0 
    - Is transmitted over wires continuously
    - Transmission is effectively instant 
    - Implies that a wire contains 1 value at a time
  - ![image-20220421155017122](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625821.png)

  - Grouping
    - ![image-20220421155055397](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625822.png)
  - Circuit Delay
    - Circuit delays are fact of life
    - ![image-20220421155123871](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625823.png)
- Synchronous Digital Systems are made up of two basic types of circuits:
  - Combinational Logic (CL) circuits
    - Stateless Combinational Logic (&,|,~)
    - Output is a function of the inputs only. 
    - Similar to a pure function in mathematics, y = f(x). (No way to store information from one invocation to the next. No side effects)
  - State Elements
    - State circuits (e.g., registers)
    - circuits that store information.
- State
  - Uses for State Elements
    - As a place to store values for some indeterminate amount of time: 
    - Register files (like x0-x31 on the RISC-V)
    - Memory (caches, and main memory)
    - Help control the flow of information between combinational logic blocks.
      - State elements are used to hold up the movement of information at the inputs to combinational logic blocks and allow for orderly passage.
  - Register is used to hold up the transfer of data to adder.
    - ![image-20220422132019015](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625824.png)

- Register Details…What’s inside? **Flip-Flop**
  - ![image-20220422133855653](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625825.png)
- What’s the timing of a Flip-flop?
  - q quiescent
  - ![image-20220422134108478](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625826.png)
  - ![image-20220422135851707](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625827.png)
- Accumulator proper timing
  - ![image-20220422141824912](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625828.png)
  - ![image-20220422141904516](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625829.png)

- Maximum Clock Frequency
  - ![image-20220422142526701](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625830.png)

- Pipelining to improve performance
  - ![image-20220422142658116](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625831.png)
  - ![image-20220422142911613](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625832.png)

- Timing Terms
  - Clock (CLK) - steady square wave that synchronizes system
  - Setup Time - when the input must be stable before the rising edge of the CLK
  - Hold Time - when the input must be stable after the rising edge of the CLK
  - “CLK-to-Q” Delay - how long it takes the output to change, measured from the rising edge of the CLK
  - Flip-flop - one bit of state that samples every rising edge of the CLK (positive edge-triggered)
  - Register - several bits of state that samples on rising edge of CLK or on LOAD (positive edge-triggered)

- Finite State Machines (FSM)
  - The function can be represented with a “state transition diagram”
  - With combinational logic and registers, any FSM can be implemented in hardware.
  - ![image-20220422144227105](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625833.png)
  - FSM to detect the occurrence of 3 consecutive 1’s in the input.
    - Initial state: double circle or arrow in
    - ![image-20220422144404204](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625834.png)

- Hardware Implementation of FSM
  - … Therefore a register is needed to hold the representation of which state the machine is in.
    - Use a unique bit pattern for each state.
  - Combinational logic circuit is used to implement a function mapping the input and present state (PS) input to the next state (NS) and output.
  - ![image-20220422145314953](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625835.png)
  - Hardware for FSM: Combinational Logic
    - ![image-20220422145532167](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625836.png)

- General Model for Synchronous Systems
  - ![image-20220422145929269](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625837.png)
- Design Hierarchy
  - ![image-20220422150204849](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625838.png)

- In conclusion
  - State elements are used to: 
    - Build memories 
    - Control the flow of information between other state elements and combinational logic
  - D-flip-flops used to build registers
  - Clocks tell us when D-flip-flops change
    - setup and hold times are important
  - We pipeline long-delay CL for faster clock
  - Finite state machines extremely useful
    - You’ll see them again 151A, 152, 164, 172, …

- Combinational Logic
  - Truth Tables
    - ![image-20220422151051919](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625839.png)
  - 1 iff one (not both) a,b=1
    - ![image-20220422151617049](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625840.png)
  - 2-bit adder
    - How Many Rows? 2^4
  - 32-bit unsigned adder
    - How Many Rows? 2^64
  - 3-input majority circuit
- Logic Gates
  - ![image-20220422155328619](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625841.png)
  - ![image-20220422155443684](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625842.png)
  - 2-input gates extend to n-inputs
    - **XOR is a 1 iff the # of 1s at its input is odd**
  - Truth Table to Gates 
    - ![image-20220422160437956](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625843.png)

- Boolean Algebra
  - Power of Boolean Algebra
    - there’s a one-to-one correspondence between circuits made up of AND(+), OR(·) and NOT(/) gates and equations in BA
- Laws of Boolean Algebra
  - ![image-20220422185006235](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625844.png)

- Canonical forms
  - Sum-of-products (ORs of ANDs)

- In conclusion
  - Pipeline big-delay CL for faster clock
  - Finite State Machines extremely useful
  - You’ll see them again in (at least) 151A, 152 & 164
  - Use this table and techniques we learned to transform from 1 to another
    - ![image-20220422194225962](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625845.png)

- Combinational Logic Blocks

  - Data Multiplexer

    - 2-to-1, n-bit-wide
      - ![image-20220422194512386](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625846.png)
    - How do we build a 1-bit-wide mux?
      - ![image-20220422194655983](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625848.png)
    - 4-to-1 Multiplexor?
      - ![image-20220422195001277](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625849.png)

    - Mux: is there any other way to do it?
      - Ans: Hierarchically!
      - ![image-20220422195120235](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625850.png)

  - Arithmetic and Logic Unit
    - Most processors contain a special logic block called  Arithmetic and Logic Unit” (ALU)
    - We’ll show you an easy one that does ADD, SUB, bitwise AND (&), bitwise OR (|)
    - ![image-20220422195835337](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625851.png)
    - Our simple ALU
      - ![image-20220422200003306](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625852.png)

- Adder / Subtracter Design

  - Truth-table, then determine canonical form, then minimize and implement as we’ve seen before
  - Look at breaking the problem down into smaller pieces that we can cascade or hierarchically layer
  - One-bit adder
    - ![image-20220423100231538](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625853.png)
  - N 1-bit adders → 1 N-bit adder
    - ![image-20220423100452612](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625854.png)
      - Overflow for unsigned
      - Signed ?
    - Sum of two 2-bit numbers
      - ![image-20220423101147581](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625855.png)

  - Subtractor: A-B = A + (-B)
    - ![image-20220423102651358](https://raw.githubusercontent.com/Christina0031/Images/main/202303241625856.png)
    - SUB as $C_0$ and input of conditional inverter
    - $C_n$ Unsigned overflow flag

- In conclusion

  - Use muxes to select among input
    - S input bits selects 2^S^ inputs
    - Each input can be n-bits wide, indep of S
  - Can implement muxes hierarchically
  - ALU can be implemented using a mux
    - Coupled with basic block elements
  - N-bit adder-subtractor done using N 1bit adders with XOR gates on input
    - XOR serves as conditional inverter
  

## RISC-V Processor Design

- The CPU
  - Processor (CPU): the active part of the computer that does all the work **(data manipulation and decision-making)**
  - Datapath: portion of the processor that **contains hardware necessary to perform operations required by the processor (the brawn)**
  - Control: portion of the processor (also in hardware) that **tells the datapath what needs to be done (the brain)**
- One-Instruction-Per-Cycle RISC-V Machine
- Five Stages of the Datapath
- Basic Phases of Instruction Execution
  - ![image-20220424190043705](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645023.png)
- Datapath Components
  - Combinational elements
  - Storage elements + clocking methodology
- Each instruction during execution reads and updates the state of : (1) Registers, (2) Program counter, (3) Memory
- Complete RV32I Datapath!
  - ![image-20220424220550158](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645024.png)
  - We have designed a complete datapath
    - Capable of executing all RISC-V instructions in one cycle each
    - Not all units (hardware) used by all instructions
  - 5 Phases of execution
    - IF, ID, EX, MEM, WB
    - Not all instructions are active in all phases
  - Controller specifies how to execute instructions
    - We still need to design it
- Control and Status Registers
  - Control and status registers (CSRs) are separate from the register file (x0-x31)
    - Used for monitoring the status and performance
    - There can be up to 4096 CSRs
  - Not in the base ISA, but almost mandatory in every implementation
    - ISA is modular
    - Necessary for counters and timers, and communication with peripherals
- CSR Instructions
  - The CSRRW (Atomic Read/Write CSR) instruction ‘atomically’ swaps values in the CSRs and integer registers.
  - CSRRW reads the previous value of the CSR and writes it to integer register rd. Then writes rs1 to CSR
  - Hint: Use write enable and clock
- System Instructions
  - ecall – (I-format) makes requests to supporting execution environment (OS), such as system calls (syscalls)
  - ebreak – (I-format) used e.g. by debuggers to transfer control to a debugging environment
  - fence – sequences memory (and I/O) accesses  as viewed by other threads or co-processors

- Instruction Timing
- Control Realization Options
  - ROM
    - Read-Only Memory
    - Regular structure
    - Can be easily reprogrammed
      - fix errors
      - add instructions
    - Popular when designing control logic manually
  - Combinatorial Logic
    - Today, chip designers use logic synthesis tools to convert truth tables to networks of gates
- RV32I, A Nine-Bit ISA!
  - Instruction type encoded using only 9 bits
- ROM-based Control
  - ![image-20220425121814668](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645025.png)
- Call home, we’ve made HW/SW contact!

## Pipelining

- “Iron Law” of Processor Performance

  - ![image-20220425130606060](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645026.png)
  - Instructions per Program Determined by
    - Task
    - lgorithm, e.g. O(N2) vs O(N)
    - Programming language 
    - Compiler
    - Instruction Set Architecture (ISA)

  - (Average) Clock Cycles per Instruction (CPI) Determined by
    - ISA
    - Processor implementation (or microarchitecture) 
    - E.g. for “our” single-cycle RISC-V design, CPI = 1
    - Complex instructions (e.g. strcpy), CPI >> 1
    - Superscalar processors, CPI < 1 (next lectures)

  - Time per Cycle (1/Frequency) Determined by
    - Processor microarchitecture (determines critical path through logic gates)
    - Technology (e.g. 5nm versus 28nm)
    - Power budget (lower voltages reduce transistor speed)
- Energy per Task
  - ![image-20220425134503476](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645027.png)
- Energy Tradeoff Example
  - Significantly improved energy efficiency thanks to
    - Moore’s Law
    - Reduced supply voltage
- End of Scaling
  - In recent years, industry has not been able to reduce supply voltage much, as reducing it further would mean increasing “leakage power” where transistor switches don’t fully turn off (more like dimmer switch than on-off switch)
  - Also, size of transistors and hence capacitance, not shrinking as much as before between transistor generations
    - Need to go to 3D
  - Power becomes a growing concern – the “power wall”
- Energy “Iron Law”
  - ![image-20220425183625643](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645028.png)
  - Energy efficiency (e.g., instructions/Joule) is key metric in all computing devices
  - For power-constrained systems (e.g., 20MW datacenter), need better energy efficiency to get more performance at same power
  - For energy-constrained systems (e.g., 1W phone), need better energy efficiency to prolong battery life
- Sequential Laundry
  - Pipelining doesn’t help latency of single task, it helps throughput of entire workload
  - Multiple tasks operating simultaneously using different resources
  - Potential speedup = Number of pipe stages
  - Time to “fill” pipeline and time to “drain” it reduces speedup
    - 2.3X v. 4X in this example
  - Pipeline rate limited by slowest pipeline stage
  - Unbalanced length of pipe stages reduce speedup
- Pipelined RISC-V Datapath
  - ![image-20220425195817453](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645030.png)
- Single-Cycle RV32I Datapath
  - ![image-20220426214705821](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645031.png)

- Pipelined RV32I Datapath
  - ![image-20220426215731199](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645032.png)

- Pipelining Hazards
  - A hazard is a situation that prevents starting the next instruction in the next clock cycle
  - Structural hazard
    - A required resource is busy (e.g. needed in multiple stages)
  - Data hazard
    - Data dependency between instructions
    - Need to wait for previous instruction to complete its data read/write
  - Control hazard
    - Flow of execution depends on previous instruction

- Structural Hazard
  - Problem: Two or more instructions in the pipeline compete for access to a single physical resource
  - Solution 1: Instructions take it in turns to use resource, some instructions have to stall
  - Solution 2: Add more hardware to machine
  - Can always solve a structural hazard by adding more hardware

- Regfile Structural Hazards
  - Each instruction:
    - Can read up to two operands in decode stage
    - Can write one value in writeback stage
  - Avoid structural hazard by having separate “ports”
    - Two independent read ports and one independent write port
  - Three accesses per cycle can happen simultaneously

- Structural Hazard: Memory Access
  - Instruction and Data Caches
  - Fast, on-chip memory, separate for instructions and data

- Structural Hazards – Summary
  - Conflict for use of a resource
  - In RISC-V pipeline with a single memory
    - Load/store requires data access
    - Without separate memories, instruction fetch would have to stall for that cycle
      - All other operations in pipeline would have to wait
  - Pipelined datapaths require separate instruction/data memories
    - Or separate instruction/data caches
  - RISC ISAs (including RISC-V) designed to avoid structural hazards
    - e.g. at most one memory access/instruction

- Data Hazard
  - Solution 1: Stalling
    - Bubble:
      - Effectively nop: Affected pipeline stages do “nothing”
  - Solution 2: Forwarding
    - Use result when it is computed
      - Don’t wait for it to be stored in a register
      - Requires extra connections in the datapath
    - Data Needed for Forwarding
      - Compare destination of older instructions in pipeline with sources of new instruction in decode stage.
      - Must ignore writes to x0!
- Stalls and Performance
  - Stalls reduce performance
    - But stalls are required to get correct results
  - Compiler can arrange code or insert nops (addi x0, x0, 0) to avoid hazards and stalls
    - Requires knowledge of the pipeline structure

- Forwarding (aka Bypassing)
  - Use result when it is computed
    - Don’t wait for it to be stored in a register
    - Requires extra connections in the datapath
  - Use result when it is computed
  - Don’t wait for it to be stored in a register
  - Requires extra connections in the datapath

- Load Data Hazard
  - Slot after a load is called a **load delay slot**
    - If that instruction uses the result of the load, then the hardware will stall for one cycle
    - Equivalent to inserting an explicit nop in the slot
      - except the latter uses more code space
    - Performance loss
  - Idea: 
    - Put unrelated instruction into load delay slot
    - No performance loss!

- Control Hazards
  - If branch not taken, then instructions fetched sequentially after branch are correct
  - If branch or jump taken, then need to flush incorrect instructions from pipeline by
    converting to NOPs
- Reducing Branch Penalties
  - Every taken branch in simple pipeline costs 2 dead cycles
  - To improve performance, use “branch prediction” to guess which way branch will go earlier in pipeline
  - Only flush pipeline if branch prediction was incorrect
  
- Increasing Processor Performance

  - Clock rate
    - Limited by technology and power dissipation

  - Pipelining
    - “Overlap” instruction execution
    - Deeper pipeline: 5 => 10 => 15 stages
      - Less work per stage
      - shorter clock cycle
      - But more potential for hazards (CPI > 1)

  - Multi-issue “superscalar” processor

## Caches

- Memory Hierarchy
  - If level closer to Processor, it is:
    - Smaller
    - Faster
    - More expensive
    - subset of lower levels (contains most recently used data)
  - Lowest Level (usually disk=HDD/SSD) contains all available data
  - Memory Hierarchy presents the processor with the illusion of a very large & fast memory

- Direct-Mapped Cache
  - In a direct-mapped cache, each memory address is associated with one possible block within the cache
  - ![image-20220429164434046](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645033.png)
  - ![image-20220428221823622](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645034.png)
  - ![image-20220429165026175](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645035.png)

- In the previous problem, we had a Direct-Mapped cache, in which blocks map to specifically one slot in our cache. This is good for quick replacement and finding out block, but **not good for efficiency of space**!
- We define **associativity** as the number of slots a block can potentially map to in our cache. Thus, a Fully-Associative cache has the most associativity, meaning every block can go anywhere in the cache.
- Fully Associative Cache
  - Tag: same as before 
  - Offset: same as before
  - Index: non-existant
  - ![image-20220429165300743](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645036.png)

- N-Way Set Associative Cache 

  - Tag: same as before
  - Offset: same as before
  - Index: points us to the correct “**row**” (called a
    **set** in this case)

  - So what’s the difference?
    - each set contains multiple blocks
    - once we’ve found correct set, must compare with all tags in that set to find our data
    - **Size of $ is # sets × N blocks/set × block size**

  - In fact, for a cache with M blocks,
    - it’s Direct-Mapped if it’s 1-way set assoc
    - it’s Fully Assoc if it’s M-way set assoc 
    - so these two are just special cases of the more general set associative design
  - ![image-20220429234408009](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645037.png)
  
- Average Memory Access Time
  - Hit Time + Miss Penalty x Miss Rate
  - I didn't even care what the hit rate is. Whether it's a hit or miss, I'm paying the hit time.
- Analyzing Multi-level cache hierarchy
  - ![image-20220430105230057](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645038.png)

- Hit and Miss Policies

  - Here’s a reminder about the three different cache hit policies you should know about:

    - **Write-back** means that on a write hit, data is written to the cache only, and when this write happens, the dirty bit for the block that was written becomes 1. Writing to the cache is fast, so write latency in write-back caches is usually quite small. However, when a block is evicted from a write-back cache, if the dirty bit is 1, memory must be updated with the contents of that block, as it contains changes that are not yet reflected in memory. This makes write-back caches more difficult to implement in hardware.
    - **Write-through** means that on a write hit, data is written to both the cache and main memory. Writing to the cache is fast, but writing to main memory is slow; this makes write latency in write-through caches slower than in a write-back cache. However, write-through caches mean simpler hardware, since we can assume in write-through caches that memory always has the most up-to-date data.
    - **Write-around** means that in every situation, data is written to main memory only; if we have the block we’re writing in the cache, the valid bit of the block is changed to invalid. Essentially, there is no such thing as a write hit in a write-around cache; a write “hit” does the same thing as a write miss.

    - There are also two miss policies you should know about:

      - **Write-allocate** means that on a write miss, you pull the block you missed on into the cache. For write-back, write-allocate caches, this means that memory is never written to directly; instead, writes are always to the cache and memory is updated upon eviction.

      - **No write-allocate** means that on a write miss, you do not pull the block you missed on into the cache. Only memory is updated.

    - Additionally, in this course, we talk about several replacement policies, in order from most useful to least useful (normally):

      - **LRU** - Least recently used—when we decide to evict a cache block to make space, we select the block that has been used farthest back in time of all the blocks.

      - **Random** - When we decide to evict a cache block to make space, we randomly select one of the blocks in the cache to evict.

      - **MRU** - Most recently used—when we decide to evict a cache block to make space, we select the block that has been used most recently of all the blocks.

  - Memory Access with Cache

    - ![image-20220501224516469](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645039.png)
    - https://www.cs.umb.edu/cs641/notes10.html https://www.cs.umb.edu/cs641/notes09.html
    - Cache miss is “lost time” to the system, counted officially as “CPU time” since it’s handled completely by the CPU. The CPU just rates as slower because of all the cache misses.
    - Although wait for i/o is not lost time because the OS reschedules, the wait time to access memory on a cache miss is lost. It’s too short a time to be recovered by rescheduling, which itself takes a microsecond or so. The OS is not involved at all in the operation of a cache miss if the data is in main memory. (It does get involved if a disk access is needed, but that’s a different subject.)

## 	Operating Systems

- Adding I/O
  - ![image-20220430155857731](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645040.png)

- What Does the Core of the OS Do?
  - Provide isolation between running processes
  - Provide interaction with the outside world
- What Does OS Need from Hardware?
  - Memory translation
  - Protection and privilege
    - Split the processor into at least two running modes: "User" and "Supervisor"
    - RISC-V also has "Machine" below "Supervisor"
  - Traps & Interrupts
    - A way of going into Supervisor mode on demand

## Virtual Memory

- Virtual vs. Physical Addresses
  - ![image-20220501120536476](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645041.png)

- Responsibilities of Memory Manager
  - Map virtual to physical addresses 
  - Protection
  - Swap memory to disk
- Paged Memory Address Translation
  - ![image-20220501140318942](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645042.png)

- Page Table Stored in Memory
  - ![image-20220501144802452](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645043.png)
  - Two (slow) memory accesses per lw/sw on cache miss
  - How could we minimize the performance penalty?
    - Transfer blocks (not words) between DRAM and processor cache
      - Exploit spatial locality
    - Use a cache for frequently used page table entries …

- Blocks vs. Pages
  - 16 KiB DRAM, 4 KiB Pages (for VM), 128 B blocks (for caches), 4B words (for lw/sw)

- Memory Access
  - ![image-20220501150032874](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645044.png)

- Hierarchical Page Table

  - ![image-20220501152055905](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645045.png)

  - 32-b RISC-V
    - ![image-20220501152449796](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645046.png)

- Translation Lookaside Buffers (TLB)
  - Where Are TLBs Located?
    - ![image-20220501181139833](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645047.png)

- Address Translation Using TLB
  - ![image-20220501181619217](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645048.png)

- Page-Based Virtual-Memory Machine
  - ![image-20220501183121552](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645049.png)
- Address Translation
  - ![image-20220501183146375](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645050.png)

## I/O

- Adding I/O
  - ![image-20220430155857731](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645040.png)

- Instruction Set Architecture for I/O
  - What must the processor do for I/O?
    - Input: Read a sequence of bytes
    - Output: Write a sequence of bytes
  - Interface options: Memory mapped I/O
    - Portion of address space dedicated to I/O
    - I/O device registers there (no memory)
    - Use normal load/store instructions, e.g. lw/sw
    - Very common, used by RISC-V

- If a program polls a device say every second, and does something else in the mean time if no data is available (including possibly just sleeping, leaving the CPU available for others), it's **polling**.
- If the program continuously polls the device (or resource or whatever) without doing anything in between checks, it's called a busy-wait.

- Polling: Processor Checks Status, Then Acts
  - Device registers generally serve two functions: 
    - Control Register, says it’s OK to read/write (I/O ready)
    - Data Register, contains data
  - Processor reads from Control Register in loop
    - Waiting for device to set Ready bit in Control reg (0 → 1)
    - Indicates “data available” or “ready to accept data”
  - Processor then loads from (input) or writes to
    (output) data register 
    - I/O device resets control register bit (1 → 0)
  - Procedure called “Polling”

-  Interrupts and DMA
  - Low data rate (e.g. mouse, keyboard)
    - Use interrupts.
    - Overhead of interrupts ends up being low
  - High data rate (e.g. network, disk)
    - Start with interrupts...
      - If there is no data, you don't do anything!
    - Once data starts coming... Switch to Direct Memory Access (DMA)

- Direct Memory Access (DMA)
  - DMA engine contains registers written by CPU: 
    - Memory address to place data
    - \# of bytes 
    - I/O device #, direction of transfer
    - unit of transfer, amount to transfer per burst
  - ![image-20220501203407295](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645051.png)

## Parallelism

- Choice of hardware and software parallelism are independent 
  - Concurrent software can also run on serial hardware
  - Sequential software can also run on parallel hardware
- Flynn’s Taxonomy
  - Flynn’s Taxonomy is for parallel hardware
  - Single Instruction/Single Data Stream (SISD)
    - ![image-20220502002012948](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645052.png)
  - Single Instruction/Multiple Data Stream (SIMD)
    - ![image-20220502002039917](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645053.png)
  - Multiple Instruction/Multiple Data Stream (MIMD)
    - ![image-20220502002110474](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645054.png)
  - Multiple Instruction/Single Data Stream (MISD)
    - ![image-20220502002145274](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645055.png)

- Intel SSE Intrinsics
  - Intrinsics are C functions and procedures for putting in assembly language, including SSE instructions
  - With intrinsics, can program using these instructions indirectly
  - One-to-one correspondence between SSE instructions and intrinsics

## Thread-Level Parallelism

- Multiprocessor Execution Model
  - Each processor (core) executes its own instructions
  - Separatere sources (not shared) 
    - Datapath (PC, registers, ALU) 
    - Highest level caches (e.g., 1st and 2nd)
  - Shared resources
    - Memory (DRAM) 
    - Often 3rd level cache  
      - Often on same silicon chip
      - But not a requirement
  - Shared memory 
    - Each “core” has access to the entire memory in the processor
    - Special hardware keeps caches consistent (next lecture!)
    - Advantages:   
      - Simplifies communication in program via shared variables
    - Drawbacks:   
      - Does not scale well:
        - “Slow” memory shared by many “customers” (cores) 
        - May become bottleneck (Amdahl’s Law)
  - Two ways to use a multiprocessor:
    - Job-level parallelism 
      - Processors work on unrelated problems   
      - No communication between programs
    - Partition work of single task between several cores
      - E.g., each performs part of large matrix multiplication

- Operating System Threads
  - Give illusion of many “simultaneously” active threads 
  - Multiplex software threads onto hardware threads:
    - Switch out blocked threads (e.g., cache miss, user input, network access)
    - Timer (e.g., switch active thread every 1 ms)
  - Remove a software thread from a hardware thread by 
    - Interrupting its execution 
    - Saving its registers and PC to memory 
  - Start executing a different software thread by
    - Loading its previously saved registers into a hardware thread’s registers
    - Jumping to its saved PC

- Hardware Assisted Software Multithreading
  - ![image-20220503142751268](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645056.png)
- Hyper-Threading
  - Simultaneous Multithreading (HT): Logical CPUs > Physical CPUs
  - Run multiple threads at the same time per core 
  - Each thread has own architectural state (PC, Registers, etc.)
  - Share resources (cache, instruction unit, execution units)

- Thread: sequence of instructions, with own program counter and processor state (e.g., register file)
- Multicore:   
  - Physical CPU: One thread (at a time) per CPU, in software OS switches threads typically in response to I/O events like disk read/write
  - Logical CPU: Fine-grain thread switching, in hardware, when thread blocks due to cache miss/memory access
  - Hyper-Threading aka Simultaneous Multithreading (SMT): Exploit superscalar architecture to launch instructions from different threads at the same time!

- Multiprocessor Caches
  - ![image-20220503180757618](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645057.png)

- Memory access to cache is either
  - ![image-20220503183056757](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645058.png)

## MapReduce & Spark

- Amdahl’s Law
  - ![image-20220503190733946](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645059.png)
- MapReduce
  - Simple data-parallel programming model designed for scalability and fault-tolerance
  - MapReduce is a wonderful abstraction for programming thousands of machines
  - Hides details of machine failures, slow machines
  - File-based
- Spark does it even better
  - Memory-based
  - Lazy evaluation

## Datacenters & Cloud Computing

- Defining Performance
  - Response Time or Latency
    - time between start and completion of a task
  - Throughput or Bandwidth
    - total amount of work in a given time

- Coping with Performance in Array
  - ![image-20220504103915490](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645060.png)

- Impact of latency, bandwidth, failure, varying workload on WSC software?
- Overall WSC Energy Efficiency: amount of computational work performed divided by the total energy used in the process
- Power Usage Effectiveness (PUE): Total building power / IT equipment power

## Dependability via Redundancy

- Fault: failure of a component
  - May or may not lead to system failure

- Spatial Redundancy – replicated data or check information or hardware to handle hard and soft (transient) failures
- Temporal Redundancy – redundancy in time (retry) to handle soft (transient) failures

- Dependability Measures
- Availability Measures
- Reliability Measures
  - Failures In Time (FIT) Rate

- Error Detection/Correction Codes

- Hamming came up with simple to understand mapping to allow Error Correction at minimum distance of three
  - Single error correction, double error detection
  - Hamming ECC
  - https://en.wikipedia.org/wiki/ECC_memory
  - Interleave data and parity bits
  - ![image-20220504132716518](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645061.png)

- What if More Than 2-Bit Errors?
  - ![image-20220504132850090](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645062.png)

- RAID: Redundant Arrays of (Inexpensive) Disks 
- And in Conclusion
  - Great Idea: Redundancy to Get Dependability
    - Spatial (extra hardware) and Temporal (retry if error)
  - Reliability: MTTF, Annualized Failure Rate (AFR), and FIT 
  - Availability: % uptime (MTTF/MTTF+MTTR)
  - Memory 
    - Hamming distance 2: Parity for Single Error Detect 
    - Hamming distance 3: Single Error Correction Code + encode bit position of error
  - Treat disks like memory, except you know when a disk has failed—erasure makes parity an Error Correcting Code
  - RAID-2, -3, -4, -5 (and -6, -10): Interleaved data and parity

## GPU

- CPU vs GPU

  - ![image-20220504135514711](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645063.png)

  - ![image-20220504135626263](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645064.png)

- GPU
  - ![image-20220504140016155](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645065.png)

- Graphics Pipeline

  - ![image-20220504140407535](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645066.png)
  - ![image-20220504140722224](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645067.png)
  - ![image-20220504141021365](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645068.png)

  - ![image-20220504141321636](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645069.png)
  - ![image-20220504141358240](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645070.png)
  - ![image-20220504141414303](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645072.png)
  - ![image-20220504141441336](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645073.png)
  - ![image-20220504141532592](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645074.png)
  - ![image-20220504141553744](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645075.png)
  - ![image-20220504141611861](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645076.png)

- ![image-20220504141819249](https://raw.githubusercontent.com/Christina0031/Images/main/202303241645077.png)

