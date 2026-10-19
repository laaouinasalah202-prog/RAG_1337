import fire


class Cli:
    def __init__(
        self,
        max_chunk_size=2000,
        k=5,
        dataset_path="data.json",
        save_directory="cache",
        student_search=None,
        student_search_results_path="results.json",
    ):
        self.max_chunk_size = max_chunk_size
        self.k = 60
        self.dataset_path = dataset_path
        self.save_directory = save_directory
        self.student_search = student_search
        self.student_search_results_path = student_search_results_path

    def index(self):
        pass

    def search(self):
        pass

    def search_dataset(self):
        pass

    def answear(self):
        pass

    def answer_dataset(self):
        pass

    def evaluate(self):
        pass


if __name__ == "__main__":
    fire.Fire(Cli)