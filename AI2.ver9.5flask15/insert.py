import os

# フォルダのパスを指定してください
folder_path = "C:\\Users\\223204\\Desktop\\all\\emotion\\train\\Dog\\angry"

# フォルダ内のファイルを取得
for file_name in os.listdir(folder_path):
    # ファイルのフルパスを取得
    full_path = os.path.join(folder_path, file_name)
    
    # ファイルかどうかを確認
    if os.path.isfile(full_path):
        # 新しいファイル名を作成
        new_file_name = "angry-dog-" + file_name
        new_full_path = os.path.join(folder_path, new_file_name)
        
        # ファイル名を変更
        os.rename(full_path, new_full_path)
        print(f"{file_name} を {new_file_name} に変更しました。")

print("すべての画像ファイル名の変更が完了しました。")
