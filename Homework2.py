p = 7
q = 13
e = 5

n = p*q
phi = (p-1)*(q-1)
d = 29

m = "jayhawk"

def toInt(m):
    tmp = []
    for x in range(len(m)):
        tmp.append(ord(m[x])-96)
    return tmp

def toChar(m):
    tmp = []
    for x in range(len(m)):
        tmp.append(chr(m[x]+96))
    return ''.join(tmp)

def publicKey(m, e=5, n=91):
    c = pow(m,e) % n
    return c

def privateKey(c, d=29, n=91):
    m = pow(c,d) % n
    return m

print("Public Key: <", e, ":", n, ">")
print("Private Key: <", d, ":", n, ">")

intInput = toInt(m)

print("Plain Text Input: ", m)
print("Integer Input: ", intInput)

crypt = []
for x in range(len(intInput)):
    crypt.append(publicKey(intInput[x]))

print("Encoded Ints: ", crypt)

plain = []
for x in range(len(crypt)):
    plain.append(publicKey(crypt[x]))

print("Decoded Ints: ", plain)

decodeText = toChar(plain)

print("Decoded Text: ", decodeText)