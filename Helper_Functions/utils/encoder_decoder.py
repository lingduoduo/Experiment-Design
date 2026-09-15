def encoder(first_strs, second_strs, separator="|", method="utf-16"):
    str_original = first_strs + separator + second_strs
    return str_original.encode(encoding=method)

def decoder(str_original_encoded, separator="|", method="utf-16"):
    str_original = str_original_encoded.decode(method)
    first_strs, second_strs = str_original.split(separator)
    return first_strs, second_strs