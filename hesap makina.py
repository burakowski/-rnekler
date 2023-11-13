def topla(x, y):
    return x + y

def cikar(x, y):
    return x - y

def carp(x, y):
    return x * y

def bol(x, y):
    return x / y

print("Lütfen yapmak istediğiniz işlemi seçin.")
print("1. Toplama")
print("2. Çıkarma")
print("3. Çarpma")
print("4. Bölme")

secim = input("Seçiminiz (1/2/3/4): ")

sayi1 = float(input("İlk sayıyı girin: "))
sayi2 = float(input("İkinci sayıyı girin: "))

if secim == '1':
    print(sayi1,"+",sayi2,"=", topla(sayi1,sayi2))

elif secim == '2':
    print(sayi1,"-",sayi2,"=", cikar(sayi1,sayi2))

elif secim == '3':
    print(sayi1,"*",sayi2,"=", carp(sayi1,sayi2))

elif secim == '4':
    if sayi2 == 0:
        print("Bir sayı sıfıra bölünemez.")
    else:
        print(sayi1,"/",sayi2,"=", bol(sayi1,sayi2))
else:
    print("Geçersiz seçim.")