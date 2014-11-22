# -*- coding: utf-8 -*-

"""A reference implementation of cubehash in pure Python.

When we say "reference implementation", we're really actually just using fancy
words to say, "Don't ever fucking use this code for anything other than
research. You may also potentially find it useful for printing out, framing
it, hanging it on the wall, and just looking at it but never executing it."

:authors: Isis <isis@torproject.org> 0xA3ADB67A2CDB8B35
:copyright: (c) 2014, Isis Agora Lovecruft
:license: see LICENSE for licensing information
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

import binascii
import math
import sys


class BadParameter(Exception):
    """An input parameter was invalid."""

class MalformedBlock(ValueError):
    """A message block was malformed."""

class UsageError(ValueError):
    """An attempt to use the API was incorrect."""


def carry_uint32(x):
    """Ensure that **x** is positive by bitwise ANDing it with the largest
    expressible 32-bit integer (0xFFFFFFFF, i.e. 2^32).

    :param int x: An integer expressible in 32-bits or less.
    :rtype: int
    :returns: The **x**, carried around the maximum positive 32-bit integer.
    """
    # We floor() because x might be a float
    return (int(math.floor(x)) & (int('0xFFFFFFFF', 16)))

def rotate_uint32(x, n):
    """Rotate a 32-bit number **n** times, keeping within 32-bit integer size.

    :param int x: The number to rotate.
    :param int n: The number of times to rotate **x**.
    """
    try:
        rotated = (int(x) << (n))
        # XXX Do we need to raise ValueError if `rotated==0` here?
        #
        # We probably should, because we want to evaluate x>>32-n if
        # rotated==0.
        if rotated == 0:
            raise ValueError
    except ValueError:  # negative shift count
        rotated = (int(x) >> (32 - (n)))

    return carry_uint32(rotated)

def count_bits(string):
    """Count the number of bits in an arbitrary string by coercing its type to
    a unicode literal, encoding that into UTF-8 so that we have a less
    inaccurate count of the # actual bytes, then multiplying the length of
    that by 8.

    :param string: A string-like thing.
    :rtype: int
    :returns: Probably¹ the correct number of bits in the **string**

    ¹ Fuck Python.
    """
    return len(encode_utf8(string)) * 8

def encode_utf8(string, enforce="strict"):
    """Encode a string into UTF-8.

    :param string: A string-like thing.
    :rtype: str
    :returns: The **string** encoded into UTF-8.
    """
    return type('')(string).encode('utf-8', enforce)

def binary(byte):
    """Convert a single byte into its binary representation.

    This turns out to be more complicated that it should be in Python,
    because, as I've already said, fuck Python.

    If the input **byte** were, for example, ``'a'``, this would return the
    binary form. THOUGH NOT THE WAY PYTHON DOES IT, because Python is
    seriously braindamaged. So, if you did this the dumb/Python way, you might
    be tempted to do:

        >>> bin(ord('a'))
        '0b1100001'

    But this is a string. Not binary. Another braindamaged Python
    string. Actually, if you're a smart Python programmer, and also incredibly
    lucky, you might get a unicode. Which is almost equally braindamaged.

    Thus, we do it this way:

        >>> int(format(ord('a'), 'b'))
        1100001
        >>> int(format(2, 'b'))
        10
        >>> int(format(13, 'b'))
        1101

    Voilá.

    >>> binary('a')
    1100001
    >>> binary(2)
    10
    >>> binary(2.0)
    10
    >>> binary(0b10)
    10
    >>> binary('0b10')
    10
    >>> binary(0x02)
    10
    >>> binary('0x02')
    10
    >>> binary('\x02')
    10

    :type byte: bytes, str, unicode, or int
    :param byte: A byte.
    :raises ValueError: if we cannot format **byte** as binary.
    :rtype: Python doesn't have a type for this.
    :returns: A fucking goddamned binary representation, fo' realz.
    """
    try:
        # Evaluates to True for integers in any of the following forms:
        # 2, 2.0, 0b11, and 0x2. This is *not* True for the string form '0b11'.
        if isinstance(byte, int):
            return     int(format(byte,              'b'))
        elif isinstance(byte, float):
            return     int(format(int(byte),         'b'))
        elif ((sys.version_info.major == 2 and isinstance(byte, (basestring, unicode))) or
              (sys.version_info.major == 3 and isinstance(byte, str))):
            if byte.startswith('0b'):
                return int(format(int(byte,      2), 'b'))
            elif byte.startswith('0x'):
                return int(format(int(byte,     16), 'b'))
            elif ord(byte) == 128:  # u'\x80'
                return int(format(ord(byte),         'b'))
            else:
                return int(format(ord(byte),         'b'))
        elif byte is None:
            return     int(format(    0,             'b'))
        else:
            raise ValueError("Can't format `%r` as binary" % byte)
    except ValueError as error:
        print(str(error))
        raise ValueError("Can't format `%r` as binary" % byte)


class CubeHashState(object):
    """The internal state of a hash digest algorith."""

    #: The 128-bit state is viewed as a sequence of 32 4-byte words, each of
    #: which is interpreted in little-endian form as a 32-bit integer (ranging
    #: from [0, 31] inclusive).
    _state = [ 0b00000, 0b00001, 0b00010, 0b00011,
               0b00100, 0b00101, 0b00110, 0b00111,
               0b01000, 0b01001, 0b01010, 0b01011,
               0b01100, 0b01101, 0b01110, 0b01111,
               0b10000, 0b10001, 0b10010, 0b10011,
               0b10100, 0b10101, 0b10110, 0b10111,
               0b11000, 0b11001, 0b11010, 0b11011,
               0b11100, 0b11101, 0b11110, 0b11111, ]

    def __init__(self, algorithm):
        """Initialize the state.

        :type algorithm: :class:`CubeHash`
        :param algorithm: The overarching :class:`CubeHash` instance.
        """
        self._initialized = False
        self._finalized = False
        self.algorithm = algorithm

    def __len__(self):
        return len(self._state)

    def __getitem__(self, key):
        return self._state[key]

    def __setitem__(self, key, value):
        self._state[key] = value

    def __delitem__(self, key):
        pass

    def __iter__(self):
        iter(self._state)

    def __reversed__(self):
        self._state.reverse()

    def __contains__(self, item):
        return item in self._state

    def __missing__(self, key):
        raise ValueError("%s only contains 32 4-byte integers, not %r."
                         % (self.__class__.__name__, key))

    def __str__(self):
        s  = "[ %s, %s, " % (str(self._state[0]).zfill(10),  str(self._state[1]).zfill(10))
        s += "%s, %s,\n"  % (str(self._state[2]).zfill(10),  str(self._state[3]).zfill(10))
        s += "  %s, %s, " % (str(self._state[4]).zfill(10),  str(self._state[5]).zfill(10))
        s += "%s, %s,\n"  % (str(self._state[6]).zfill(10),  str(self._state[7]).zfill(10))
        s += "  %s, %s, " % (str(self._state[8]).zfill(10),  str(self._state[9]).zfill(10))
        s += "%s, %s,\n"  % (str(self._state[10]).zfill(10), str(self._state[11]).zfill(10))
        s += "  %s, %s, " % (str(self._state[12]).zfill(10), str(self._state[13]).zfill(10))
        s += "%s, %s,\n"  % (str(self._state[14]).zfill(10), str(self._state[15]).zfill(10))
        s += "  %s, %s, " % (str(self._state[16]).zfill(10), str(self._state[17]).zfill(10))
        s += "%s, %s,\n"  % (str(self._state[18]).zfill(10), str(self._state[19]).zfill(10))
        s += "  %s, %s, " % (str(self._state[20]).zfill(10), str(self._state[21]).zfill(10))
        s += "%s, %s,\n"  % (str(self._state[22]).zfill(10), str(self._state[23]).zfill(10))
        s += "  %s, %s, " % (str(self._state[24]).zfill(10), str(self._state[25]).zfill(10))
        s += "%s, %s,\n"  % (str(self._state[26]).zfill(10), str(self._state[27]).zfill(10))
        s += "  %s, %s, " % (str(self._state[28]).zfill(10), str(self._state[29]).zfill(10))
        s += "%s, %s ]\n" % (str(self._state[30]).zfill(10), str(self._state[31]).zfill(10))
        return s

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.algorithm)

    def _initialize(self):
        """Initialise the :data:`state`.

        From http://cubehash.cr.yp.to/ :
           |
           | CubeHash produces the initial state as follows. Set the first
           | three state words x[00000], x[00001], x[00010] to the integers
           | h/8, b, r respectively. Set the remaining state words to 0. Then
           | transform the state invertibly through i rounds. Of course, the
           | implementor can eliminate these transformations at the expense of
           | storage by precomputing the initial state for any particular
           | h,b,r.
           |
        """
        if self._initialized:
            return

        # Initialise the first three words:
        self._state[0] = self.algorithm.output_bits / 8
        self._state[1] = self.algorithm.blocksize
        self._state[2] = self.algorithm.rounds

        # Set the remaining state words to 0:
        for word in range(3, 32):
            self._state[word] = 0

        if self.algorithm._debug: print(self._state)

        # Transform the state invertibly through i rounds:
        for index in range(0, self.algorithm.initialization_rounds):
            self.algorithm._round()

        self._initialized = True

    def _finalize(self):
        """Finalize the :data:`state`.

        From http://cubehash.cr.yp.to/ :
           |
           | After all input blocks are handled, CubeHash produces the final
           | hash as follows. Xor the integer 1 into the last state word
           | x[11111]. Transform the state invertibly through f
           | rounds. Finally, output the first h/8 bytes of the state.
           |
        """
        if self._finalized:
            return

        # Xor the integer 1 into the last state word x[11111]:
        self._state[31] ^= 1

        # Transform the state invertibly through f rounds:
        for index in range(0, self.algorithm.finalization_rounds):
            self.algorithm._round()

        size = int(self.algorithm.output_bits / 8)  # typically 64 bytes
        size = int(size / 4)  # each state word is 4 bytes
        self.algorithm._hash = self._state[:size]

        self._finalized = True


class CubeHash(object):
    """A reference implementation of `CubeHash`_.

    .. _CubeHash: http://cubehash.cr.yp.to/
    """
    #: A parameter i in {1,2,3,...}, the number of initialization rounds,
    #: typically 16.
    initialization_rounds=16

    #: A parameter f in {1,2,3,...}, the number of finalization rounds,
    #: typically 32.
    finalization_rounds = 32

    #: A parameter r in {1,2,3,...}, the number of rounds per message block,
    #: typically 16.
    rounds = 16

    #: A parameter b in {1,2,3,...,128}, the number of bytes per message
    #: block, typically 32.
    blocksize = 32

    #: A parameter h in {8,16,24,...,512}, the number of output bits,
    #: typically 512.
    output_bits = 512

    def __init__(self, message=None):
        """Digest a **message** with CubeHash.

        :type message: str or None
        :param message: DOCDOC
        """
        #: (unicode) A message m, a string of bits between 0 bits and 2^128-1
        #: bits, encoded into UTF-8.
        self._message = type('')()
        #: (int) The calculated number of bits in the message.
        self._message_bits = 0
        #: (unicode) The final output hash, assigned once all rounds have been
        #: completed.
        self._hash = type('')()

        #: If ``True``, print debugging messages to stdout.
        self._debug = True

        self.state = CubeHashState(self)

        if message:
            self.hash(message)

    def __check_message__(self, message, enforce="strict"):
        """The **message** should be a string between 0 bits and 2^128-1 bits.

        :raises BadParameter: if the **message** is invalid.
        :param str message: The message to check for conformity to the
            parameter constraints.
        :param str enforce: The option to pass to ``encode()`` to specify the
            strictness with which we should enforce encoding to UTF-8.
        :rtype: bool
        :returns: ``True`` if the message is well-formed. ``False`` if you
            somehow missed all of the errors raised, and yet still somehow did
            not return ``True``.
        """
        if not (((sys.version_info.major == 2) and isinstance(message, basestring)) or
                ((sys.version_info.major == 3) and isinstance(message, str))):
            raise BadParameter("The message must be a string.")

        self._message = encode_utf8(message, enforce=enforce)
        self._message_bits = count_bits(message)

        if not 0 <= self._message_bits <= (2**128 - 1):
            raise BadParameter("message must be between 0 and 2^128-1 bits.")
        else:
            return True
        return False

    def _pad_message(self, message):
        """Pad the message to a multiple of ``8 * blocksize``.

        From http://cubehash.cr.yp.to/ :
           |
           | The message is first padded to create a sequence of b-byte input
           | blocks. Padding works as follows: append a 1 bit; then append the
           | minimum possible number of 0 bits to reach a multiple of 8b
           | bits. (The bits in a byte are first 128, then 64, then 32, then
           | 16, then 8, then 4, then 2, then 1.) Implementations restricted
           | to byte-aligned messages can simply append a 128 byte and then
           | the minimum possible number of 0 bytes to reach a multiple of b
           | bytes.
           |

        :param str message: The message to pad before hashing.
        :rtype: unicode
        :returns: The padded message.
        """
        if not (self._message_bits and self._message):
            self.__check_message__(message)

        padded = self._message
        # The message should be padded to a multiple of 8*blocksize:
        if not (len(padded) % self.blocksize == 0):
            # Append a byte with bits set to 128 (0b10000000, i.e. '\x80') first:
            padded += u'\x80'
            while (len(padded) % self.blocksize != 0):
                # Because Python is restrictively byte-aligned, we'll just take
                # DJB's shortcut and use all zeroes from here on out:
                padded += u'\x00'

        return padded

    def _get_blocks(self, message):
        """Chunk a message into blocks :data:`blocksize` bytes long.

        :param message: A message string which has already been padded,
            i.e. with :meth:`_pad_message`.
        :raises StopIteration: Once all the bytes of the message are consumed.
        :rtype: tuple
        :yields: A variable number of tuples. Each tuple will contain
            *exactly* :data:`blocksize` elements, one element for each
            byte.
        """
        blocks = []

        for i in range(0, len(message), self.blocksize):
            yield message[i:i+self.blocksize]

    def _round(self, block=None):
        """Put a message block through a round of the CubeHash algorithm.

        Algorithm:
        ~~~~~~~~~~
        From http://cubehash.cr.yp.to/ :
           |
           | CubeHash maintains a 128-byte state. It xors the first b-byte
           | input block into the first b bytes of the state, transforms the
           | state invertibly through r identical rounds, xors the next b-byte
           | input block into the first b bytes of the state, transforms the
           | state invertibly through r identical rounds, xors the next b-byte
           | input block into the first b bytes of the state, transforms the
           | state invertibly through r identical rounds, etc.
           |
           | The 128-byte state is viewed as a sequence of 32 4-byte words
           | x[00000], x[00001], x[00010], x[00011], x[00100], ..., x[11111],
           | each of which is interpreted in little-endian form as a 32-bit
           | integer.
           |
           | A round has the following ten steps:
           |
           | 1. Add x[0jklm] into x[1jklm] modulo 2^32, for each (j,k,l,m).
           | 2. Rotate x[0jklm] upwards by 7 bits, for each (j,k,l,m).
           | 3. Swap x[00klm] with x[01klm], for each (k,l,m).
           | 4. Xor x[1jklm] into x[0jklm], for each (j,k,l,m).
           | 5. Swap x[1jk0m] with x[1jk1m], for each (j,k,m).
           | 6. Add x[0jklm] into x[1jklm] modulo 2^32, for each (j,k,l,m).
           | 7. Rotate x[0jklm] upwards by 11 bits, for each (j,k,l,m).
           | 8. Swap x[0j0lm] with x[0j1lm], for each (j,l,m).
           | 9. Xor x[1jklm] into x[0jklm], for each (j,k,l,m).
           | 10. Swap x[1jkl0] with x[1jkl1], for each (j,k,l).
           |
           | That's it.
           |

        Efficiency:
        ~~~~~~~~~~~
        From http://cubehash.cr.yp.to/ :
           |
           | Overall a round has 32 32-bit additions and 32 32-bit xors, so
           | CubeHashr/b has 32r/b 32-bit additions and 32r/b 32-bit xors for
           | each byte of the padded message; in other words, 128r/b bit
           | additions and 128r/b bit xors for each bit of the padded
           | message. The finalization has 32f 32-bit additions and 32f 32-bit
           | xors, comparable cost to handling fb/r bytes of input. The
           | initialization, if not precomputed, has 32i 32-bit additions and
           | 32i 32-bit xors, comparable cost to handling ib/r bytes of input.
           |

        :type block: list or None
        :param block: A chunk of the message to hash, exactly
            :data:`blocksize` bytes in length (default: 32 bytes). The
            **block** can be ``None`` (and should be ``None``) during
            initialization and finalization rounds.
        :raises MalformedBlock: If the **block** is the wrong size.
        """
        if block is not None:
            if len(block) != self.blocksize:
                raise MalformedBlock(("A message block had the wrong size: %d "
                                      "bytes. Expected size: %d bytes.")
                                     % (len(block), self.blocksize))

            # 0. Initialise the state vector.
            for word in range(0, 32):
                #self.state[word] ^= int(str(binary(block[word])), 2)
                self.state[word] ^= ord(block[word])

        # 1. Add x[0jklm] into x[1jklm] modulo 2^32, for each (j,k,l,m).
        #----------------------------------------------------------------------
        # The following does not work, mostly because Python bit vectors and
        # bitwise operations aren't native types. :(
        #
        # for word in range(0, 32):
        #     word = binary(word)  ## binary(word)<<1 → {0,2,20,22,200,202,…,}
        #----------------------------------------------------------------------
        # Results in tuples:
        #      {(0,16),            (1,17),            …, (15,31)}
        # i.e. {(0b00000,0b10000), (0b00001,0b10001), …, (0b01111,0b11111)}
        for zero_index, one_index in zip(range(0, 16), range(16, 32)):
            self.state[one_index] += self.state[zero_index]
            self.state[one_index]  = carry_uint32(self.state[one_index])

        # 2. Rotate x[0jklm] upwards by 7 bits, for each (j,k,l,m).
        for zero_index in range(0, 16):
            self.state[zero_index] = rotate_uint32(self.state[zero_index], 7)

        # 3. Swap x[00klm] with x[01klm], for each (k,l,m).
        for zero_zero_index, zero_one_index in zip(range(0, 8), range(8, 16)):
            self.state[zero_zero_index], self.state[zero_one_index] = \
            self.state[zero_one_index], self.state[zero_zero_index]

        # 4. Xor x[1jklm] into x[0jklm], for each (j,k,l,m).
        for zero_index, one_index in zip(range(0, 16), range(16, 32)):
            self.state[zero_index] ^= self.state[one_index]

        # 5. Swap x[1jk0m] with x[1jk1m], for each (j,k,m).
        #                      {0b10000,…,0b11101} = {16,…,29}
        #                                     {0b10010,…,0b11111} = {18,…,31}
        for zero_secondary, one_secondary in zip(range(16, 30), range(18, 32)):
            self.state[zero_secondary], self.state[one_secondary] = \
            self.state[one_secondary], self.state[zero_secondary]

        # 6. Add x[0jklm] into x[1jklm] modulo 2^32, for each (j,k,l,m).
        for zero_index, one_index in zip(range(0, 16), range(16, 32)):
            self.state[one_index] += self.state[zero_index]
            self.state[one_index]  = carry_uint32(self.state[one_index])

        # 7. Rotate x[0jklm] upwards by 11 bits, for each (j,k,l,m).
        for zero_index in range(0, 16):
            self.state[zero_index] = rotate_uint32(self.state[zero_index], 11)

        # 8. Swap x[0j0lm] with x[0j1lm], for each (j,l,m).
        for zero_tertiary, one_tertiary in zip(range(0, 12), range(4, 16)):
            self.state[zero_tertiary], self.state[one_tertiary] = \
            self.state[one_tertiary], self.state[zero_tertiary]

        # 9. Xor x[1jklm] into x[0jklm], for each (j,k,l,m).
        for zero_index, one_index in zip(range(0, 16), range(16, 32)):
            self.state[zero_index] ^= self.state[one_index]

        # 10. Swap x[1jkl0] with x[1jkl1], for each (j,k,l).
        for zero_primary, one_primary in zip(range(16, 31), range(17, 32)):
            self.state[zero_primary], self.state[one_primary] = \
            self.state[one_primary], self.state[zero_primary]

        if self._debug: print(self.state)

    def hash(self, message):
        """Produce a CubeHash digest of **message**.

        :param str message: The message to hash.
        """
        self.state._initialized = False
        self.state._finalized = False

        if self._debug: print("Initializing state...")
        self.state._initialize()

        if self._debug: print("Encoding and padding message...")
        self._message = self._pad_message(message)

        if self._debug:
            print(("\nHashing message: %r\n"
                   "Message length:  %d bytes\n"
                   "Padded length:   %d bytes\n")
                  % (message, len(message), len(self._message)))
            print("Hashing message blocks...")

        for block in self._get_blocks(self._message):
            if self._debug: print("Processing message block %r." % block)
            self._round(block)
            for round in range(0, self.rounds - 1):
                self._round()

        if self._debug: print("\nFinalizing state...")
        self.state._finalize()

        if self._debug: print("Done hashing message.")

    def _get_digest(self):
        """Return the CubeHash digest in hexadecimal format.

        :rtype: int
        :returns: A base-10 integer for the digested message.
        """
        if not self._hash:
            raise UsageError("You must hash() a message first!")

        digest = type('')()

        for x in self._hash:
            # XXX
            #digest += type('')(x).zfill(10)  # pad with 0s to len(str(2**32))
            #digest += type('')(x)
            #string = type('')(x)
            #digest += type('')(chr(int(string[:5])))
            #digest += type('')(chr(int(string[5:])))
            string = type('')(x).zfill(10)  # pad with 0s to len(str(2**32))
            digest += type('')(binary(string))

        return int(digest)

    def digest(self):
        """Return the CubeHash digest in hexadecimal format.

        :rtype: long
        :returns: A long representing the digested message.
        """
        return long(self._get_digest())

    def hexdigest(self):
        """Return the CubeHash digest in hexadecimal format.

        :rtype: str
        :returns: A hexadecimal string for the digested message.
        """
        #return hex(self._get_digest()).replace('0x', '').replace('L', '')
        return binascii.hexlify(str(self._get_digest()).replace('L', ''))


if __name__ == "__main__":
    # Test vector relevant to later implementation of CubeHash as a circuit in
    # R1CS (libsnark):
    cube = CubeHash
    cube.output_bits = 256
    cube.blocksize = 64
