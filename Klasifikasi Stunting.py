import pandas as pd
from tkinter import Tk, filedialog, Label, Button, messagebox
from tkinter.ttk import Combobox
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import matplotlib.pyplot as plt

le_gender = LabelEncoder()
le_age = LabelEncoder()
le_weight = LabelEncoder()
le_height = LabelEncoder()
le_target = LabelEncoder()

def import_data():
    global df
    try:
        file_path = filedialog.askopenfilename(
            title="Pilih File",
            filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv")]
        )
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise ValueError("Format file tidak didukung!")

        df['Jenis Kelamin'] = df['Jenis Kelamin'].replace({'L': 'Laki-laki', 'P': 'Perempuan', 0: 'Perempuan', 1: 'Laki-laki'})
        df['Target'] = df['Target'].replace({0: 'Normal', 1: 'Stunting', 'Normal': 'Normal', 'Stunting': 'Stunting'})
        df['Kelas Umur'] = df['Umur'].apply(lambda x: 'Bayi' if x < 12 else 'Balita')

        def bb_category(row):
            if row['Kelas Umur'] == 'Bayi':
                return 'Rendah' if row['Berat Badan'] < 6 else ('Normal' if row['Berat Badan'] <= 10 else 'Lebih')
            else:
                return 'Rendah' if row['Berat Badan'] < 9 else ('Normal' if row['Berat Badan'] <= 13 else 'Lebih')

        def tb_category(row):
            if row['Kelas Umur'] == 'Bayi':
                return 'Pendek' if row['Tinggi Badan'] < 65 else ('Normal' if row['Tinggi Badan'] <= 75 else 'Tinggi')
            else:
                return 'Pendek' if row['Tinggi Badan'] < 75 else ('Normal' if row['Tinggi Badan'] <= 87 else 'Tinggi')

        df['Kelas Berat Badan'] = df.apply(bb_category, axis=1)
        df['Kelas Tinggi Badan'] = df.apply(tb_category, axis=1)

        df['Target'] = df.apply(
            lambda row: 'Stunting' if row['Kelas Berat Badan'] == 'Rendah' and row['Kelas Tinggi Badan'] == 'Pendek'
            else ('Normal' if row['Kelas Berat Badan'] in ['Normal', 'Lebih'] and row['Kelas Tinggi Badan'] in ['Normal', 'Tinggi'] else row['Target']),
            axis=1
        )

        df['Jenis Kelamin'] = le_gender.fit_transform(df['Jenis Kelamin'])
        df['Kelas Umur'] = le_age.fit_transform(df['Kelas Umur'])
        df['Kelas Berat Badan'] = le_weight.fit_transform(df['Kelas Berat Badan'])
        df['Kelas Tinggi Badan'] = le_height.fit_transform(df['Kelas Tinggi Badan'])
        df['Target'] = le_target.fit_transform(df['Target'])

        print("Distribusi Target untuk Lebih + Tinggi:")
        print(df[(df['Kelas Berat Badan'] == le_weight.transform(['Lebih'])[0]) & 
                 (df['Kelas Tinggi Badan'] == le_height.transform(['Tinggi'])[0])]['Target'].value_counts())

        df_laki = df[df['Jenis Kelamin'] == le_gender.transform(['Laki-laki'])[0]]
        df_perempuan = df[df['Jenis Kelamin'] == le_gender.transform(['Perempuan'])[0]]

        df_laki_normal = df_laki[df_laki['Target'] == le_target.transform(['Normal'])[0]]
        df_laki_stunting = df_laki[df_laki['Target'] == le_target.transform(['Stunting'])[0]]
        if len(df_laki_stunting) < len(df_laki_normal):
            df_laki_stunting_upsampled = resample(df_laki_stunting, replace=True, n_samples=len(df_laki_normal), random_state=42)
            df_laki = pd.concat([df_laki_normal, df_laki_stunting_upsampled])
        else:
            df_laki_normal_upsampled = resample(df_laki_normal, replace=True, n_samples=len(df_laki_stunting), random_state=42)
            df_laki = pd.concat([df_laki_stunting, df_laki_normal_upsampled])

        df_perempuan_normal = df_perempuan[df_perempuan['Target'] == le_target.transform(['Normal'])[0]]
        df_perempuan_stunting = df_perempuan[df_perempuan['Target'] == le_target.transform(['Stunting'])[0]]
        if len(df_perempuan_stunting) < len(df_perempuan_normal):
            df_perempuan_stunting_upsampled = resample(df_perempuan_stunting, replace=True, n_samples=len(df_perempuan_normal), random_state=42)
            df_perempuan = pd.concat([df_perempuan_normal, df_perempuan_stunting_upsampled])
        else:
            df_perempuan_normal_upsampled = resample(df_perempuan_normal, replace=True, n_samples=len(df_perempuan_stunting), random_state=42)
            df_perempuan = pd.concat([df_perempuan_stunting, df_perempuan_normal_upsampled])

        df = pd.concat([df_laki, df_perempuan]).sample(frac=1, random_state=42).reset_index(drop=True)

        messagebox.showinfo("Sukses", "Data berhasil diimpor, diseimbangkan, dan diproses!")
    except Exception as e:
        messagebox.showerror("Error", f"Terjadi kesalahan: {e}")

def train_model():
    global model
    try:
        X = df[['Jenis Kelamin', 'Kelas Umur', 'Kelas Berat Badan', 'Kelas Tinggi Badan']]
        y = df['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = MultinomialNB()
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test)) * 100
        messagebox.showinfo("Model Dilatih", f"Model berhasil dilatih dengan akurasi: {acc:.2f}%")
    except Exception as e:
        messagebox.showerror("Error", f"Kesalahan saat melatih model: {e}")

def predict():
    try:
        jk = gender_combobox.get()
        umur = age_combobox.get()
        bb = weight_combobox.get()
        tb = height_combobox.get()

        jk_encoded = le_gender.transform([jk])[0]
        umur_encoded = le_age.transform([umur])[0]
        bb_encoded = le_weight.transform([bb])[0]
        tb_encoded = le_height.transform([tb])[0]

        input_data = pd.DataFrame([
            [jk_encoded, umur_encoded, bb_encoded, tb_encoded]
        ], columns=['Jenis Kelamin', 'Kelas Umur', 'Kelas Berat Badan', 'Kelas Tinggi Badan'])

        print(f"Input: Jenis Kelamin={jk} ({jk_encoded}), Kelas Umur={umur} ({umur_encoded}), "
              f"Kelas Berat Badan={bb} ({bb_encoded}), Kelas Tinggi Badan={tb} ({tb_encoded})")

        if bb == "Rendah" and tb == "Pendek":
            result = "Stunting"
        elif (bb in ["Normal", "Lebih"] and tb in ["Normal", "Tinggi"]):
            result = "Normal"
        else:
            prediction = model.predict(input_data)[0]
            result = le_target.inverse_transform([prediction])[0]

        print(f"Hasil Prediksi: {result}")
        messagebox.showinfo("Hasil Prediksi", f"Hasil Prediksi: {result}")
    except Exception as e:
        messagebox.showerror("Error", f"Terjadi kesalahan: {e}")

def predict_from_excel():
    try:
        file_path = filedialog.askopenfilename(
            title="Pilih File untuk Prediksi",
            filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv")]
        )
        if file_path.endswith('.xlsx'):
            input_df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            input_df = pd.read_csv(file_path)
        else:
            raise ValueError("Format file tidak didukung!")

        required_columns = ['Jenis Kelamin', 'Umur', 'Berat Badan', 'Tinggi Badan']
        if not all(col in input_df.columns for col in required_columns):
            raise ValueError("Kolom yang diperlukan tidak lengkap!")

        input_df['Jenis Kelamin'] = input_df['Jenis Kelamin'].apply(lambda x: 'Perempuan' if x in ["P", "Perempuan", 0] else 'Laki-laki')
        input_df['Jenis Kelamin'] = le_gender.transform(input_df['Jenis Kelamin'])
        input_df['Kelas Umur'] = input_df['Umur'].apply(lambda x: 'Bayi' if x < 12 else 'Balita')
        input_df['Kelas Umur'] = le_age.transform(input_df['Kelas Umur'])

        def calculate_bb(row):
            if row['Kelas Umur'] == le_age.transform(['Bayi'])[0]:
                return le_weight.transform(['Rendah'])[0] if row['Berat Badan'] < 6 else (le_weight.transform(['Normal'])[0] if row['Berat Badan'] <= 10 else le_weight.transform(['Lebih'])[0])
            else:
                return le_weight.transform(['Rendah'])[0] if row['Berat Badan'] < 9 else (le_weight.transform(['Normal'])[0] if row['Berat Badan'] <= 13 else le_weight.transform(['Lebih'])[0])

        def calculate_tb(row):
            if row['Kelas Umur'] == le_age.transform(['Bayi'])[0]:
                return le_height.transform(['Pendek'])[0] if row['Tinggi Badan'] < 65 else (le_height.transform(['Normal'])[0] if row['Tinggi Badan'] <= 75 else le_height.transform(['Tinggi'])[0])
            else:
                return le_height.transform(['Pendek'])[0] if row['Tinggi Badan'] < 75 else (le_height.transform(['Normal'])[0] if row['Tinggi Badan'] <= 87 else le_height.transform(['Tinggi'])[0])

        input_df['Kelas Berat Badan'] = input_df.apply(calculate_bb, axis=1)
        input_df['Kelas Tinggi Badan'] = input_df.apply(calculate_tb, axis=1)

        input_df['Prediksi'] = input_df.apply(
            lambda row: 'Stunting' if row['Kelas Berat Badan'] == le_weight.transform(['Rendah'])[0] and row['Kelas Tinggi Badan'] == le_height.transform(['Pendek'])[0]
            else ('Normal' if row['Kelas Berat Badan'] in [le_weight.transform(['Normal'])[0], le_weight.transform(['Lebih'])[0]] and 
                           row['Kelas Tinggi Badan'] in [le_height.transform(['Normal'])[0], le_height.transform(['Tinggi'])[0]] else None),
            axis=1
        )

        mask = input_df['Prediksi'].isna()
        if mask.any():
            predictions = model.predict(input_df.loc[mask, ['Jenis Kelamin', 'Kelas Umur', 'Kelas Berat Badan', 'Kelas Tinggi Badan']])
            input_df.loc[mask, 'Prediksi'] = le_target.inverse_transform(predictions)

        save_path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                                 filetypes=[("Excel files", "*.xlsx")],
                                                 title="Simpan Hasil Prediksi")
        input_df.to_excel(save_path, index=False)

        # Dekode kolom untuk visualisasi
        input_df['Jenis Kelamin'] = le_gender.inverse_transform(input_df['Jenis Kelamin'])
        input_df['Kelas Umur'] = le_age.inverse_transform(input_df['Kelas Umur'])
        input_df['Kelas Berat Badan'] = le_weight.inverse_transform(input_df['Kelas Berat Badan'])
        input_df['Kelas Tinggi Badan'] = le_height.inverse_transform(input_df['Kelas Tinggi Badan'])

        # Grafik 1: Distribusi Normal vs Stunting berdasarkan Jenis Kelamin
        counts_gender = input_df.groupby(['Jenis Kelamin', 'Prediksi']).size().unstack(fill_value=0)
        fig1, ax1 = plt.subplots()
        counts_gender.plot(kind='bar', ax=ax1, color=['#36A2EB', '#FF6384'])
        ax1.set_title('Distribusi Prediksi Berdasarkan Jenis Kelamin')
        ax1.set_xlabel('Jenis Kelamin')
        ax1.set_ylabel('Jumlah')
        ax1.legend(title='Prediksi')
        plt.show()

        # Grafik 2: Distribusi Normal vs Stunting berdasarkan Kelas Umur
        counts_age = input_df.groupby(['Kelas Umur', 'Prediksi']).size().unstack(fill_value=0)
        fig2, ax2 = plt.subplots()
        counts_age.plot(kind='bar', ax=ax2, color=['#36A2EB', '#FF6384'])
        ax2.set_title('Distribusi Prediksi Berdasarkan Kelas Umur')
        ax2.set_xlabel('Kelas Umur')
        ax2.set_ylabel('Jumlah')
        ax2.legend(title='Prediksi')
        plt.show()

        # Grafik 3: Distribusi Kelas Berat Badan
        counts_weight = input_df['Kelas Berat Badan'].value_counts()
        fig3, ax3 = plt.subplots()
        ax3.pie(counts_weight, labels=counts_weight.index, autopct='%1.1f%%', colors=['#FF6384', '#36A2EB', '#FFCE56'])
        ax3.set_title('Distribusi Kelas Berat Badan dari Prediksi')
        plt.show()

        # Grafik 4: Distribusi Kelas Tinggi Badan
        counts_height = input_df['Kelas Tinggi Badan'].value_counts()
        fig4, ax4 = plt.subplots()
        ax4.pie(counts_height, labels=counts_height.index, autopct='%1.1f%%', colors=['#FF6384', '#36A2EB', '#FFCE56'])
        ax4.set_title('Distribusi Kelas Tinggi Badan dari Prediksi')
        plt.show()

        # Grafik 5: Distribusi Prediksi
        counts_target = input_df['Prediksi'].value_counts()
        fig5, ax5 = plt.subplots()
        ax5.pie(counts_target, labels=counts_target.index, autopct='%1.1f%%', colors=['#36A2EB', '#FF6384'])
        ax5.set_title('Distribusi Prediksi (Normal vs Stunting)')
        plt.show()

        messagebox.showinfo("Sukses", "Prediksi selesai! Hasil disimpan di file yang dipilih dan grafik distribusi ditampilkan.")
    except Exception as e:
        messagebox.showerror("Error", f"Terjadi kesalahan: {e}")

# GUI
root = Tk()
root.title("Klasifikasi Stunting dengan Naive Bayes")
root.geometry("375x530")

Label(root, text="Aplikasi Klasifikasi Stunting", font=("Arial", 16, "bold")).grid(row=0, column=0, columnspan=2, pady=20, padx=10, sticky="nsew")
Button(root, text="Pilih File Data", command=import_data).grid(row=1, column=0, columnspan=2, pady=10, padx=20, sticky="ew")
Button(root, text="Latih Model", command=train_model).grid(row=2, column=0, columnspan=2, pady=5, padx=20, sticky="ew")

Label(root, text="Jenis Kelamin :", font=("Arial", 12)).grid(row=3, column=0, pady=5, padx=20)
gender_combobox = Combobox(root, values=["Laki-laki", "Perempuan"])
gender_combobox.grid(row=3, column=1, pady=5, padx=20, sticky="w")

Label(root, text="Kelas Umur :", font=("Arial", 12)).grid(row=4, column=0, pady=5, padx=20)
age_combobox = Combobox(root, values=["Bayi", "Balita"])
age_combobox.grid(row=4, column=1, pady=5, padx=20, sticky="w")
Label(root, text="<12 bulan = Bayi", fg="grey").grid(row=5, column=0, columnspan=1, padx=20)
Label(root, text="12–24 bulan = Balita", fg="grey").grid(row=6, column=0, columnspan=1, padx=20)

Label(root, text="Kelas Berat Badan :", font=("Arial", 12)).grid(row=7, column=0, pady=5, padx=20, sticky="e")
weight_combobox = Combobox(root, values=["Rendah", "Normal", "Lebih"])
weight_combobox.grid(row=7, column=1, pady=5, padx=20, sticky="w")
Label(root, text="Bayi: <6 kg = Rendah, 6–10 kg = Normal, >10 kg = Lebih", fg="grey").grid(row=8, column=0, columnspan=2, padx=20)
Label(root, text="Balita: <9 kg = Rendah, 9–13 kg = Normal, >13 kg = Lebih", fg="grey").grid(row=9, column=0, columnspan=2, padx=20)

Label(root, text="Kelas Tinggi Badan :", font=("Arial", 12)).grid(row=10, column=0, pady=5, padx=20, sticky="e")
height_combobox = Combobox(root, values=["Pendek", "Normal", "Tinggi"])
height_combobox.grid(row=10, column=1, pady=5, padx=20, sticky="w")
Label(root, text="Bayi: <65 cm = Pendek, 65–75 cm = Normal, >75 cm = Tinggi", fg="grey").grid(row=11, column=0, columnspan=2, padx=20)
Label(root, text="Balita: <75 cm = Pendek, 75–87 cm = Normal, >87 cm = Tinggi", fg="grey").grid(row=12, column=0, columnspan=2, padx=20)

Button(root, text="Prediksi", command=predict).grid(row=14, column=0, columnspan=2, pady=(20, 5), padx=20, sticky="ew")
Button(root, text="Prediksi dari File", command=predict_from_excel).grid(row=15, column=0, columnspan=2, pady=10, padx=20, sticky="ew")

root.mainloop()