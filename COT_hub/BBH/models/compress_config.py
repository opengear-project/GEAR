class GPT2CompressConfig(dict):
    def __init__(
        self,
        compress_method=None,
        attention_number=12,
        quantize_bit=0,
        group_num=0,
        rank=0.0,
        rankv=0.0,
        loop=0,
        top_k=0.0,
        left=0.0,
        stage=1,
        device_num=0,
        batch_num=1,
        start_saving=0.0,
        locality_saving=0.0,
        token_preserving=False,
        streaming=False,
        streaming_gap=0,
        iter=0,
        # h2o setings
        heavy_size=0,
        recent_size=0,
    ):
        self.compress_method = compress_method
        self.quantize_bit = quantize_bit
        self.group_num = group_num
        self.rank = rank
        self.rankv = rankv
        self.ranv = rankv
        self.loop = loop
        self.device_num = device_num
        self.attention_number = attention_number
        self.top_k = top_k
        self.left = left
        self.batch_num = batch_num
        self.stage = stage
        self.start_saving = start_saving
        self.locality_saving = locality_saving
        self.token_preserving = token_preserving
        self.iter = iter
        self.heavy_size = heavy_size
        self.recent_size = recent_size
        self.streaming = streaming
        self.streaming_gap = streaming_gap

    def create_attention_config(self, config):
        attention_config = []
        for i in range(self.attention_number):
            attention_config.append(config)
        return attention_config

    def copy_for_all_attention(self):
        self.compress_method = self.create_attention_config(self.compress_method)
        self.quantize_bit = self.create_attention_config(self.quantize_bit)
        self.group_num = self.create_attention_config(self.group_num)
        self.rank = self.create_attention_config(self.rank)
        self.loop = self.create_attention_config(self.loop)
        self.top_k = self.create_attention_config(self.top_k)
        self.device_num = self.create_attention_config(self.device_num)
        self.left = self.create_attention_config(self.left)
        self.stage = self.create_attention_config(self.stage)
        self.rankv = self.create_attention_config(self.rankv)
        self.start_saving = self.create_attention_config(self.start_saving)
        self.locality_saving = self.create_attention_config(self.locality_saving)
        self.token_preserving = self.create_attention_config(self.token_preserving)
        self.iter = self.create_attention_config(self.iter)
        self.heavy_size = self.create_attention_config(self.heavy_size)
        self.recent_size = self.create_attention_config(self.recent_size)
        self.streaming = self.create_attention_config(self.streaming)
        self.streaming_gap = self.create_attention_config(self.streaming_gap)

    def compress_ratio(
        self,
        compress_method,
        seqlen,
        model_dim,
        rank=0,
        rankv=0,
        quantize_bit=0,
        top_k=0,
        left=0.0,
        stage=1,
        batch_num=1,
    ):
        if compress_method == None:
            return 1.0
        elif compress_method == "Picache":
            if seqlen > rank and seqlen > rankv:
                return (
                    2
                    * seqlen
                    * batch_num
                    * model_dim
                    / (
                        ((model_dim + seqlen * batch_num) * (rank + rankv))
                        * quantize_bit
                        / 16
                    )
                )
            elif seqlen <= rank:
                return (
                    (
                        2
                        * seqlen
                        * batch_num
                        * model_dim
                        / (
                            (model_dim + seqlen * batch_num) * rankv
                            + seqlen * batch_num * model_dim
                        )
                    )
                    * 16
                    / quantize_bit
                )

            elif seqlen <= rankv:
                return (
                    (
                        2
                        * seqlen
                        * batch_num
                        * model_dim
                        / (
                            (model_dim + seqlen * batch_num) * rank
                            + seqlen * batch_num * model_dim
                        )
                    )
                    * 16
                    / quantize_bit
                )
        elif compress_method == "poweriteration":
            return (
                seqlen
                * batch_num
                * model_dim
                / ((model_dim + seqlen * batch_num) * rank)
            )
        elif compress_method == "stagept":
            return (
                seqlen
                * batch_num
                * model_dim
                / (model_dim * rank + seqlen * batch_num * (rank / stage))
            )
        elif (
            compress_method == "uniformquantization"
            or compress_method == "groupquantization"
            or compress_method == "sortquantization"
        ):
            return 16 / quantize_bit
        elif compress_method == "pruning":
            return 1 / top_k
        elif (
            compress_method == "densesparseuniformquantization"
            or compress_method == "densesparsesortquantization"
        ):
            return 1 / (quantize_bit / 16 + left)
        elif compress_method == "pt+outlier":
            return (
                seqlen
                * batch_num
                * model_dim
                * 16
                / quantize_bit
                / ((model_dim + seqlen * batch_num) * rank)
            )

    def calculate_compress_ratio_list(self, seqlen, model_dim):
        self.compress_ratio_list = []
        for i, compress_method in enumerate(self.compress_method):
            if compress_method == None:
                self.compress_ratio_list.append(
                    self.compress_ratio(compress_method, seqlen, model_dim)
                )
            elif compress_method == "Picache":
                self.compress_ratio_list.append(
                    self.compress_ratio(
                        compress_method,
                        seqlen,
                        model_dim,
                        rank=self.rank[i],
                        rankv=self.rankv[i],
                        quantize_bit=self.quantize_bit[i],
                        batch_num=self.batch_num,
                        left=self.left[i],
                    )
                )
            elif compress_method == "poweriteration":
                self.compress_ratio_list.append(
                    self.compress_ratio(
                        compress_method,
                        seqlen,
                        model_dim,
                        rank=self.rank[i],
                        batch_num=self.batch_num,
                    )
                )
            elif compress_method == "stagept":
                self.compress_ratio_list.append(
                    self.compress_ratio(
                        compress_method,
                        seqlen,
                        model_dim,
                        rank=self.rank[i],
                        batch_num=self.batch_num,
                        stage=self.stage[i],
                    )
                )
            elif (
                compress_method == "uniformquantization"
                or compress_method == "groupquantization"
                or compress_method == "sortquantization"
            ):
                self.compress_ratio_list.append(
                    self.compress_ratio(
                        compress_method,
                        seqlen,
                        model_dim,
                        rank=0,
                        quantize_bit=self.quantize_bit[i],
                    )
                )
            elif compress_method == "pruning":
                self.compress_ratio_list.append(
                    self.compress_ratio(
                        compress_method,
                        seqlen,
                        model_dim,
                        top_k=self.top_k[i],
                    )
                )
            elif compress_method == "densesparseuniformquantization":
                self.compress_ratio_list.append(
                    self.compress_ratio(
                        compress_method,
                        seqlen,
                        model_dim,
                        quantize_bit=self.quantize_bit[i],
                        left=self.left[i],
                    )
                )
            elif compress_method == "densesparsesortquantization":
                self.compress_ratio_list.append(
                    self.compress_ratio(
                        compress_method,
                        seqlen,
                        model_dim,
                        quantize_bit=self.quantize_bit[i],
                        left=self.left[i],
                    )
                )
            elif compress_method == "pt+outlier":
                self.compress_ratio_list.append(
                    self.compress_ratio(
                        compress_method,
                        seqlen,
                        model_dim,
                        rank=self.rank[i],
                        quantize_bit=self.quantize_bit[i],
                        batch_num=self.batch_num,
                        left=self.left[i],
                    )
                )

    def calculate_compress_ratio_total(self):
        return sum(self.compress_ratio_list) / len(self.compress_ratio_list)

    def __str__(self):
        return f"compress_method:{self.compress_method},\nquantize_bit:{self.quantize_bit},\nrank:{self.rank},\nloop:{self.loop},\ndevice_num:{self.device_num},\ncompressratio:{self.compress_ratio_list},\ncompressratio_total:{self.calculate_compress_ratio_total()}"
