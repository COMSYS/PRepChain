import os
import csv

def enlarge_csv(input_filename, output_filename, target_size_mb):
    """
    Repeatedly write the content of input_filename to output_filename until its size reaches target_size_mb.
    Increment the ID for each row to ensure uniqueness.
    """
    # Check if the input file exists
    if not os.path.exists(input_filename):
        print(f"Error: {input_filename} does not exist.")
        return
    
    # Check if the input file is empty
    if os.path.getsize(input_filename) == 0:
        print(f"Error: {input_filename} is empty.")
        return

    # Calculate the target size in bytes
    target_size_bytes = target_size_mb * 1024 * 1024

    # Open the input and output files
    with open(input_filename, 'r', encoding='utf-8-sig') as infile, open(output_filename, 'w', encoding='utf-8-sig') as outfile:
        csvreader = csv.DictReader(infile, delimiter=";")
        csvwriter = csv.DictWriter(outfile, fieldnames=csvreader.fieldnames, delimiter=";")

        # Write the header (assuming it's the first row of the CSV)
        csvwriter.writeheader()

        # Initialize the max_id with 0
        max_id = 0

        # Write content until the file size condition is met
        while os.path.getsize(output_filename) < target_size_bytes:
            # Go to the start of the file, but skip the header
            infile.seek(0)
            next(csvreader)

            # Write rows from the input file to the output file, while updating the ID
            for row in csvreader:
                max_id += 1
                row['id'] = max_id
                csvwriter.writerow(row)

    print(f"{output_filename} has been created with a size of approximately {target_size_mb}MB.")

if __name__ == "__main__":
    # Example usage
    input_file = 'goodsreceipt_objdata.csv'   # Replace with the path to your CSV
    output_file = 'goodsreceipt5_objdata.csv'            # Name of the output file
    target_size = 5                             # Target size in MB

    enlarge_csv(input_file, output_file, target_size)
