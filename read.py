path = r'D:\0_Respaldo\000_KaggleX\Projects\Rag-Ley675\675_LAW\data\raw\Ley 675.txt'

with open(path, 'rb') as file:
    binary_data = file.read()

# Reemplazar caracteres nulos con un espacio en blanco o eliminarlos
text = binary_data.replace(b'\x00', b'').decode('utf-8', errors='replace')

print(text)
