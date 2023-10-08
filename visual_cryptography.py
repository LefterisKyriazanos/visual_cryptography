import argparse
import random
# Import for text_to_image function
from PIL import Image

parser = argparse.ArgumentParser()

# Define command line parameters
parser.add_argument('-c', type=str, default='c1', help='Cryptography method [c1, c2]')
parser.add_argument('-s', type=int, help='Number of shares to pass secret')
parser.add_argument('-d', type=str, help='Name of the output file that will contain the decrypted message')
parser.add_argument('-r', type=int, help='Seed that initializes the pseudorandom number generator')
parser.add_argument('filenames', nargs='+', help="File(name) to encrypt OR encrypted filenames to decrypt")

args = parser.parse_args()

# Rename arguments
dec_filename = args.d
method = args.c
shares = args.s 
seed = args.r 
files = args.filenames

print(args)


'''
Function from 
https://github.com/dmst-algorithms-course/assignment-2023-4/blob/main/show_txt_img.py
'''
def text_to_image(filename):
    with open(filename, 'r') as f:
        lines = [line.strip().split() for line in f]

    width = len(lines[0])
    height = len(lines)
    img = Image.new('1', (width, height))

    for y, line in enumerate(lines):
        for x, pixel in enumerate(line):
            # Invert pixel value, as 1 is black in the image.
            if pixel == '1':
                img.putpixel((x, y), 0)
            else:
                img.putpixel((x, y), 1)

    img.show()

'''
Built 2 sets of k (shares) vectors each having k elements
  - For J0: J0i = 0^𝑖10^(𝑘−𝑖) for 0 ≤ 𝑘 < 𝑘 − 1 & J0(𝑘−1) = 1^(𝑘−1)0 for 𝑘 = 𝑘 − 1
  - For J1: J1i = 0^𝑖10^(𝑘−𝑖) for 0 ≤ 𝑘 ≤ 𝑘 − 1
'''
def generate_vector_lists(k):
    vector_list_0 = [[1 if (j == i and j != k-1) else 0 for j in range(k)] for i in range(k - 1)]
    # Add the last vector of set 0 with 1s in the first (k-1) positions and 0 in the k-th position
    vector_list_0.append([1] * (k - 1) + [0])
    vector_list_1 = [[1 if j == i else 0 for j in range(k)] for i in range(k)]
    return vector_list_0, vector_list_1

'''
Creates a 2D array that has 2^k elements (lists)
Each list (element) is representing the binary form of its index
'''
def generate_binary_vectors(k):
    binary_vectors = []
    # Each of the 2^k elements represents the binary form of its index
    for i in range(2**k):
        binary_vector = [int(bit) for bit in format(i, f'0{k}b')]
        binary_vectors.append(binary_vector)

    return binary_vectors

'''
Calculates the dot product of 2 2D arrays that have k*k x k*2^k dimensions
Each dot product is calculated by performing modulo 2 operations
'''
def calculate_inner_products(vector_list, binary_vectors, k):
    final_dot_product = []
    for i in range(len(vector_list)): # k
        dot_product_row = []
        for j in range(len(binary_vectors)): # 2^k
            # calculate product in i, j position
            inner_product = 1 if 1 in [vector_list[i][m] * binary_vectors[j][m] for m in range(k)] else 0
            dot_product_row.append(inner_product)
        final_dot_product.append(dot_product_row)
   
    return final_dot_product

def parse_image_from_txt(filename, cast_to_int = False):
    try:
        # Open and read the text file
        with open(filename, 'r') as file:
            content = file.readlines()
       
        if cast_to_int:
            content = [[int(x) for x in line.strip().split()] for line in content] 
        return content
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def randomize_columns(initial_array, seed):
    # Get the number of rows and columns
    num_rows = len(initial_array)
    num_columns = len(initial_array[0])
    # Initialize the random number generator with the seed
    # If not given then seed = None
    random.seed(seed)
    # Create a list of column indices in random order
    random_column_indices = random.sample(range(num_columns), num_columns)

    # Create a new array S' with columns in random order
    s_prime = [
        [initial_array[row][random_column_indices[col]] for col in range(num_columns)]
        for row in range(num_rows)
        ]
    return s_prime

def calculate_width_height(shares, method):
    k = shares
    w = 0 
    h = 0
    if method == 'c1':
        # print('k is', k)
        w = 2 ** ((k // 2) + k % 2)
        h = 2 ** (k // 2)
    elif method == 'c2':
        w = 2 ** (((k-1)//2) + ((k-1) % 2))
        h = 2 ** ((k-1)//2)

    return w, h

'''
Creates a 2D array (list of lists) in the shape of the final share images
 - r: Number of rows of initial image (txt)
 - c: Number of columns of initial image (txt)
 - h: Number of rows in each subarray
 - w: Number of columns in each subarray
'''
def initialize_share(h, w, c= 240, r = 240):
    rows = h * r
    columns = w * c
    # Initialize the columns x rows array 
    encrypted_array = [[-1] * columns for _ in range(rows)]

    return encrypted_array

def export_array_to_txt(array, file_path = "output.txt"):
    # Open the file for writing
    with open(file_path, "w") as file:
        # Iterate through the array and write each row to the file
        for row in array:
            # Join the elements in each row as a comma-separated string
            row_str = " ".join(map(str, row))
            
            # Write the row string to the file
            file.write(row_str + "\n")
        file.close()
    print('exported ', file_path)

'''
Append a 2D array (Pixel) in desired position
 - array: list of 2D arrays
 - pixel_arr: 2D array that represents a pixel
 - share: the k-share this pixel will be added to
 - row: row index of initial image
 - row: col index of initial image
 - h: pixel height (rows)
 - w: pixel width (columns)
'''
def append_encrypted_pixel(array, pixel_arr, share, row, col, h, w):
    encrypted_array = array.copy()
    pos_i = row * h
    pos_j = col * w
    for i in range(h):
        for j in range(w):
            encrypted_array[share][pos_i + i][pos_j +j] = pixel_arr[i][j]
    return encrypted_array

def create_shares(image_rows, s0, s1, method, shares, seed, init_width, init_height):
    # Initialize row and column counters
    row = 0
    col = 0
    # S'
    s_prime = []
    # Calculate new pixel dimensions
    w, h = calculate_width_height(shares, method)
    # Initialize k (shares) 2D arrays with the appropriate shape
    encrypted_arrays = [initialize_share(h, w, init_height, init_width) for _ in range(shares)]
 
    # Iterate through each line
    for pixels in image_rows:
        # Iterate through each pixel (value)
        for pixel in pixels:
            if int(pixel) == 0:
                s_prime = randomize_columns(s0, seed)
            else: # pixel is 1
                s_prime = randomize_columns(s1, seed)
            
            # Iterate through S' index & rows
            for share, line in enumerate(s_prime):
                # Create a new array with dimensions w x h representing a pixel
                new_pixel = [[0 for _ in range(w)] for _ in range(h)]
                # Populate the new array using values from the S' k row
                for i in range(h):
                    for j in range(w):
                        new_pixel[i][j] = line[i * w + j]
                # Append encrypted pixel in final array containing all k shares
                encrypted_arrays = append_encrypted_pixel(encrypted_arrays, new_pixel, share,  row=row, col=col, w=w, h=h )
                
            col += 1  # Move to the next column
        row += 1  # Move to the next row
        col = 0  # Reset the column counter for the next row
    return encrypted_arrays


def txt_file_list_to_array(files: list()):
    encrypted_shares = [[]]
    for file_path in files:
            parsed_image = parse_image_from_txt(file_path, cast_to_int=True)
            encrypted_shares.append(parsed_image)
    # Remove initial empty item
    encrypted_shares.pop(0)
    return encrypted_shares
        
def decrypt_image_from_shares(encrypted_shares: list(list())):
    # Perform the OR operation on corresponding elements of the k arrays
    decrypted_image = []
    for i in range(len(encrypted_shares[0])):
        row = []
        for j in range(len(encrypted_shares[0][i])):
            pixel = 0
            for s in range(len(encrypted_shares)):
                if int(encrypted_shares[s][i][j]) == 1:
                    pixel = 1
            
            row.append(pixel)
        decrypted_image.append(row)
    return decrypted_image
   
'''
 - Calculates all possible subsets of the input_set
 - Splits calculated subsets to even & odd group (2D arrays)
'''
def find_subsets_with_parity(input_set):
    even_subsets = []
    odd_subsets = []

    for i in range(2 ** len(input_set)):
        # Calculate all possible subsets
        subset = [input_set[j] for j in range(len(input_set)) if (i & (1 << j)) > 0] 
        if (len(subset) % 2 == 0): # subset has even number of elements
            even_subsets.append(subset)
        else:
            odd_subsets.append(subset)

    return even_subsets, odd_subsets

def populate_s_array(subsets, shares, W):
    # Initialize the 2D list with zeros
    s = [[0 for _ in range(2**(shares-1))] for _ in range(shares)]

    # Iterate through W's k (shares) elements
    for number_index, number in enumerate(W):
        # Iterate through each of 2^(k-1) subsets
        for subset_index, subset in enumerate(subsets): 
            if (number in subset): 
                s[number_index][subset_index] = 1
    return s 

def main():
    if dec_filename != None: # Decrypt
        # Read the files and store their contents in a list of lists (k 2D arrays)
        encrypted_shares = txt_file_list_to_array(files)

        # Perform OR on the individual pixels   
        decrypted_image = decrypt_image_from_shares(encrypted_shares)
        
        # Generate decrypted image (txt)
        export_array_to_txt(decrypted_image, dec_filename)
        # optional
        text_to_image(dec_filename)
    else: # Encrypt
        rows = parse_image_from_txt(files[0], cast_to_int=True)
        # Split the file_path on the dot (.)
        image_name = files[0].split(".")[0]
        # Determine the dimensions of the input text content
        init_height = len(rows)
        init_width = len(rows[0])
        s0 = []
        s1 = []
        if method == 'c1':
            # 2D arrays (k*k)
            vector_list_0, vector_list_1 = generate_vector_lists(shares)
            # 2D array (k*2^k)
            binary_vectors = generate_binary_vectors(shares)

            # s0 & s1 are 2D arrays of k x 2^k dimensions
            s0 = calculate_inner_products(vector_list_0, binary_vectors, shares)
            s1 = calculate_inner_products(vector_list_1, binary_vectors, shares)
        
        elif method == 'c2':
            W = list(range(shares))
            # Find all even & odd subsets of W
            even_subsets, odd_subsets = find_subsets_with_parity(W)
            # s0 & s1 are 2D arrays of k x 2^(k -1) dimensions
            s0 = populate_s_array(even_subsets, shares, W)
            s1 = populate_s_array(odd_subsets, shares, W)
        
        # Calculate all k shares 
        image_shares = create_shares(rows, s0, s1, method, shares, seed, init_width, init_height)
        
        # Export k shares one by one
        for index, share in enumerate(image_shares):
            export_array_to_txt(share, file_path=f'enc_{image_name}_{index}_{method}.txt')

if __name__ == "__main__":
    main()

