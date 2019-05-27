import base64
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
from Crypto.Hash import SHA256


def generate_keypair():
    new_key = RSA.generate(4096, e=65537)
    private_key = new_key.exportKey("PEM")
    public_key = new_key.publickey().exportKey("PEM")
    return private_key, public_key


class using_key:
    def __init__(self, key_pem):
        self.key_pem = key_pem
        self._key = RSA.importKey(key_pem)
        self._key = PKCS1_OAEP.new(self._key)

    def encrypt(self, text):
        if isinstance(text, str):
            text = text.encode()
        return self._key.encrypt(text)

    def decrypt(self, ciphered_text):
        if isinstance(ciphered_text, str):
            ciphered_text = ciphered_text.encode()
        return self._key.decrypt(ciphered_text)

    def sign(self, text):
        digest = make_hash(text)
        return base64.encodebytes(self.encrypt(digest))

    def is_signature_valid(self, text, signature):
        digest = make_hash(text)
        expected_ciphered = base64.decodebytes(signature)
        expected = self.decrypt(expected_ciphered)
        return digest == expected


def make_hash(text):
    if isinstance(text, str):
        text = text.encode()
    hasher = SHA256.new()
    hasher.update(text)
    return hasher.digest()
