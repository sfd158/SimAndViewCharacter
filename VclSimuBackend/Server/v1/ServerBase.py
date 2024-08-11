'''
*************************************************************************

BSD 3-Clause License

Copyright (c) 2023,  Visual Computing and Learning Lab, Peking University

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*************************************************************************
'''

"""
Note: the asyncio library requires python >= 3.7
"""
import asyncio
import pickle
from typing import Dict, Any, List


class ServerBase:
    def __init__(self, ip_addr: str = "localhost", ip_port: int = 8888):
        self.ip_addr, self.ip_port = ip_addr, ip_port

    @property
    def size_int32(self):
        return 4

    def calc(self, mess: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    @staticmethod
    def cat_bytes(a: List[bytes]) -> bytearray:
        seq: int = sum([len(i) for i in a])
        res: bytearray = bytearray(seq)
        start: int = 0
        for i in a:
            res[start:start+len(i)] = i
            start += len(i)
        return res

    async def handle_echo(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """
        The protocal between unity client and python server is as follows:
        - 4 bytes: N bytes of message to be send
        - N bytes of message in pickle format (as pickle is convenient in python..)
        """
        print("Start Connection")
        while not reader.at_eof():
            try:
                buf_len_bytes = await reader.read(self.size_int32)
                buf_len = int.from_bytes(buf_len_bytes, "little")

                rec_buf: List[bytes] = []
                while buf_len > 0:
                    # reader.read will recieve 32768 bytes most.
                    recieve_bytes = await reader.read(buf_len)
                    rec_buf.append(recieve_bytes)
                    buf_len -= len(recieve_bytes)

                if len(rec_buf) > 0:
                    recieve_bytes = self.cat_bytes(rec_buf)
                    recieve_dict = pickle.loads(recieve_bytes)

                    dict_res = self.calc(recieve_dict)
                    mess_send = pickle.dumps(dict_res)  # what will happen if length > 32768...?
                    # print(f"send message length = {len(mess_send)}")
                    writer.write(len(mess_send).to_bytes(4, "little") + mess_send)
                    await writer.drain()

            except ConnectionError:
                break

        print('Close Connection')
        writer.close()
        self.reset()

    async def main(self):
        server = await asyncio.start_server(self.handle_echo, self.ip_addr, self.ip_port) # , limit=2097152)
        print(f'Serving on {server.sockets[0].getsockname()}')
        async with server:
            await server.serve_forever()

    def run(self):
        # Note: asyncio.run requires python 3.7+
        asyncio.run(self.main())

    async def run_in_jupyter(self):
        await self.main()


if __name__ == "__main__":
    server_base = ServerBase()
    server_base.run()
