import pandas as pd
import os

directory_path = r"C:\Users\azeem.haniffa\Desktop\Raw Mat\Pulau Meranti Dec 2024"


for file_name in os.listdir(directory_path):
    if file_name.startswith("Inventory Summary") and file_name.endswith(".xlsx"):
        input_file_path = os.path.join(directory_path, file_name)
        
        
        df = pd.read_excel(input_file_path, sheet_name="Sheet0")
        
        
        material_col = "Inventory Summary"  
        uom_col = "Unnamed: 2"  
        opening_balance_col = "Unnamed: 5"  
        
        
        df_filtered = df[[material_col, uom_col, opening_balance_col]].dropna()
        
        
        df_filtered.columns = ["Material", "UOM", "Opening Balance Total"]
        
        
        df_filtered["Concatenated"] = (
            df_filtered["Material"].astype(str) + " " +
            df_filtered["UOM"].astype(str) 
        )
        
        df_transposed = df_filtered.T
        
        
        output_file_name = file_name.replace(".xlsx", "_Transposed.csv")
        output_file_path = os.path.join(directory_path, output_file_name)
        
        df_transposed.to_csv(output_file_path, header=False, index=False)
        
        print(f"Processed data saved to {output_file_path}")
