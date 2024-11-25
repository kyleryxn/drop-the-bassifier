class FileHandler:
    @staticmethod
    def save_uploaded_file(uploaded_file, temp_file_path):
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())