import os
import sys
import re


def extract_binary_from_spaces(text):
    """
    Extract binary data from spaces after punctuation marks.
    - One space after punctuation = 0
    - Two spaces after punctuation = 1
    
    Args:
        text (str): Text containing encoded binary data
        
    Returns:
        str: Binary string (e.g., "01001010")
    """
    binary_data = []
    
    # Look for punctuation marks followed by spaces
    i = 0
    while i < len(text) - 1:
        if text[i] in ['.', ',', ';', ':']:
            # Count spaces after punctuation
            space_count = 0
            j = i + 1
            while j < len(text) and text[j] == ' ':
                space_count += 1
                j += 1
                
            # Encode: 1 space = 0, 2 spaces = 1
            if space_count == 1:
                binary_data.append('0')
            elif space_count == 2:
                binary_data.append('1')
                
            i = j - 1  # Move to the last space
        i += 1
        
    return ''.join(binary_data)

def binary_to_ascii(binary_string):
    """
    Convert a binary string to ASCII text.
    
    Args:
        binary_string (str): Binary string (e.g., "01001010")
        
    Returns:
        str: Decoded ASCII text
    """
    # Ensure binary string length is a multiple of 8
    if len(binary_string) % 8 != 0:
        print(f"Warning: Binary string length ({len(binary_string)}) is not a multiple of 8. Padding with zeros.")
        binary_string = binary_string.ljust((len(binary_string) // 8 + 1) * 8, '0')
    
    # Convert binary to ASCII
    ascii_text = ""
    for i in range(0, len(binary_string), 8):
        byte = binary_string[i:i+8]
        try:
            ascii_text += chr(int(byte, 2))
        except ValueError:
            print(f"Error converting binary '{byte}' to ASCII")
    
    return ascii_text

def decode_steganographic_message_spaces(text):
    """
    Decode a steganographic message hidden using the Punctuation Space Method.
    
    Args:
        text (str): Text containing the hidden message
        
    Returns:
        str: Decoded message
    """
    # Check for activation signal
    if "thoughtful analysis" not in text:
        print("No activation signal found. Text may not contain a hidden message.")
        return None
    
    # Extract text between activation signal and end marker
    start_marker = "thoughtful analysis"
    end_marker = "In conclusion"
    
    start_idx = text.find(start_marker)
    if start_idx == -1:
        return None
    
    # Find the end of the paragraph containing the activation signal
    paragraph_end = text.find("\n\n", start_idx)
    if paragraph_end == -1:
        paragraph_end = len(text)
    
    # Find the end marker after the activation signal
    end_idx = text.find(end_marker, paragraph_end)
    if end_idx == -1:
        end_idx = len(text)
    
    # Extract the encoded section
    encoded_section = text[paragraph_end:end_idx]
    
    # Extract binary data from spaces
    binary_data = extract_binary_from_spaces(encoded_section)
    
    # Convert binary to ASCII
    decoded_message = binary_to_ascii(binary_data)
    
    return decoded_message, binary_data

def decode_steganographic_message_firstletter(text):
    """
    Decode a steganographic message hidden using the First Word Method.
    
    Args:
        text (str): Text containing the hidden message
        
    Returns:
        str: Decoded message
    """
    # Check for activation signal
    if "interesting perspective" not in text:
        print("No activation signal found. Text may not contain a hidden message.")
        return None
    
    # Extract text between activation signal and end marker
    start_marker = "interesting perspective"
    end_marker = "Eventually"
    
    start_idx = text.find(start_marker)
    if start_idx == -1:
        return None
    
    # Find the end of the paragraph containing the activation signal
    paragraph_end = text.find("\n\n", start_idx)
    if paragraph_end == -1:
        paragraph_end = len(text)
    
    # Find the end marker after the activation signal
    end_idx = text.find(end_marker, paragraph_end)
    if end_idx == -1:
        end_idx = len(text)
    
    # Extract the encoded section
    encoded_section = text[paragraph_end:end_idx]
    
    # Split into sentences and extract first letter of first word in each sentence
    sentences = re.split(r'(?<=[.!?])\s+', encoded_section)
    hidden_message = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            words = sentence.split()
            if words:
                hidden_message += words[0][0]
    
    return hidden_message
