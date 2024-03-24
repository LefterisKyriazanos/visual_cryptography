
---

# Visual Cryptography

## Overview

This repository contains a raw implementation of the "Visual Cryptography" technique proposed by Moni Naor and Adi Shamir in 1995. Visual Cryptography is a cryptographic technique that allows for the encryption of images such that decryption can be performed visually, without the need for complex computations. The implementation is made using **only** default python libraries, without the use of ready-made graph libraries or any ready-made implementations of algorithms, or parts thereof.

## Background

In their seminal paper "Visual Cryptography," Naor and Shamir introduced a novel method for encrypting images into shares that individually reveal no information about the original image but when combined visually reveal the secret image. This technique is particularly useful for securely transmitting images over insecure channels or for storing sensitive images in a distributed manner.

## Implementation

The implementation in this repository provides a raw implementation of the Visual Cryptography technique using Python. It includes functions for encrypting an image into shares and for decrypting the shares to reveal the original image. The code is intended for educational purposes and may require further optimization for practical applications.

## Usage

To use the Visual Cryptography implementation, follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies (e.g., Python).
3. Run the `python visual_cryptography.py -s 4 laughing_smiley_bw.txt` to encrypt the image to 4 (`-s`) separate images using the first cryptography method
4. Run the `python visual_cryptography.py -d dec_laughing_smiley_bw_c1.txt
enc_laughing_smiley_bw_0_c1.txt
enc_laughing_smiley_bw_1_c1.txt
enc_laughing_smiley_bw_2_c1.txt
enc_laughing_smiley_bw_3_c1.txt` to reveal the original image out of 4 encrypted images
5. Run the `python visual_cryptography.py -c c2 -s 3 laughing_smiley_bw.txt` to encrypt the image to 3 (`-s`) separate images using the second cryptography method (more efficient)
6. Run the `python visual_cryptography.py -d dec_laughing_smiley_bw_c2.txt
enc_laughing_smiley_bw_0_c2.txt
enc_laughing_smiley_bw_1_c2.txt
enc_laughing_smiley_bw_2_c2.txt` to reveal the original image out of 3 encrypted images (encrypted with the second method)

- `-s` parameter is configurable and it represents the number of people you want to include in the image encryption/decryption.

## References

- Naor, Moni, and Adi Shamir. "Visual Cryptography." Advances in Cryptology—Eurocrypt '94. Springer, Berlin, Heidelberg, 1995.

