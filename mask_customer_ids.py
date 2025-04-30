import pandas as pd
import hashlib

def mask_customer_id(customer_id):
    """Hash the customer_id to create a masked version."""
    # Convert to string and encode
    str_id = str(customer_id).encode('utf-8')
    # Create hash
    hashed = hashlib.sha256(str_id).hexdigest()
    # Return first 12 characters of hash prefixed with 'CUST_'
    return f'CUST_{hashed[:12]}'

def mask_csv_file(input_file, output_file):
    """Mask customer IDs in a CSV file."""
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        
        # Check if customer_id column exists
        if 'customer_id' not in df.columns:
            print(f"No customer_id column found in {input_file}")
            return
        
        # Create masked version of customer_id
        df['customer_id'] = df['customer_id'].apply(mask_customer_id)
        
        # Save to new file
        df.to_csv(output_file, index=False)
        print(f"Created masked version: {output_file}")
        
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")

if __name__ == "__main__":
    # Process both files
    files_to_mask = [
        ('demo.csv', 'demo_masked.csv'),
        ('sample_demo.csv', 'sample_demo_masked.csv')
    ]
    
    for input_file, output_file in files_to_mask:
        print(f"\nProcessing {input_file}...")
        mask_csv_file(input_file, output_file) 