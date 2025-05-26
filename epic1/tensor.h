
#include <numeric>

namespace utec::algebra {

    template <typename T, size_t tam>
    class Tensor {
    private:
        std::array<size_t, tam> forma;
        std::vector<T> datos;

        template <typename... Idxs>
        size_t calculate_index(Idxs... idxs) const {
            std::array<size_t, tam> indices{static_cast<size_t>(idxs)...};
            size_t indice = 0;
            size_t paso = 1;
            for (int i = tam - 1; i >= 0; --i) {
                if (indices[i] >= forma[i]) {
                    throw std::out_of_range("índice fuera de rango");}
                indice += indices[i] * paso;
                paso *= forma[i];}
            return indice;}

        void multiply_with_broadcasting(Tensor& resultado, const Tensor& a, const Tensor& b,
                                        std::array<size_t, tam>& indices, size_t dim = 0) const {
            if (dim == tam) {
                std::array<size_t, tam> a_indices, b_indices;
                for (size_t i = 0; i < tam; ++i) {
                    a_indices[i] = (a.forma[i] == 1) ? 0 : indices[i];
                    b_indices[i] = (b.forma[i] == 1) ? 0 : indices[i];}
                resultado(indices) = a(a_indices) * b(b_indices);
                return;}
            for (size_t i = 0; i < resultado.shape()[dim]; ++i) {
                indices[dim] = i;
                multiply_with_broadcasting(resultado, a, b, indices, dim + 1);}}

    public:

        Tensor(const std::array<size_t, tam>& forma) : forma(forma) {
            size_t tam_total = 1;
            for (size_t dim : forma) tam_total *= dim;
            datos.resize(tam_total);
            std::fill(datos.begin(), datos.end(), T{});}

        template <typename... Dims>
        Tensor(Dims... dims) : forma{static_cast<size_t>(dims)...} {
            static_assert(sizeof...(Dims) == tam, "el número de dimensiones debe ser el mismo que el tamaño");
            size_t tam_total = 1;
            for (size_t dim : forma) tam_total *= dim;
            datos.resize(tam_total);
            std::fill(datos.begin(), datos.end(), T{});}

        template <typename... Idxs>
        T& operator()(Idxs... idxs) {
            return datos[calculate_index(idxs...)];}

        template <typename... Idxs>
        const T& operator()(Idxs... idxs) const {
            return datos[calculate_index(idxs...)];}

        T& operator()(const std::array<size_t, tam>& idxs) {
            return std::apply([this](auto... unpacked) -> T& {
                return (*this)(unpacked...);}, idxs);}

        const T& operator()(const std::array<size_t, tam>& idxs) const {
            return std::apply([this](auto... unpacked) -> const T& {
                return (*this)(unpacked...);}, idxs);}

        T& operator[](size_t indice) {
            if (indice >= datos.size()) throw std::out_of_range("índice fuera de rango");
            return datos[indice];}

        const T& operator[](size_t indice) const {
            if (indice >= datos.size()) throw std::out_of_range("índice fuera de rango");
            return datos[indice];}

        const std::array<size_t, tam>& shape() const noexcept {
            return forma;}

        void reshape(const std::array<size_t, tam>& nueva_forma) {
            size_t nuevo_tam = std::accumulate(nueva_forma.begin(), nueva_forma.end(), size_t{1}, std::multiplies<>());

            if (nuevo_tam != datos.size()) {
                throw std::invalid_argument("el número total de elementos debe permanecer igual");}

            auto original = forma;
            auto nuevo = nueva_forma;
            std::sort(original.begin(), original.end());
            std::sort(nuevo.begin(), nuevo.end());
            if (original != nuevo) {
                throw std::invalid_argument("la forma debe ser una permutación de las dimensiones originales");}

            forma = nueva_forma;}

        void fill(const T& valor) noexcept {
            std::fill(datos.begin(), datos.end(), valor);}

        Tensor operator+(const Tensor& otro) const {
            if (forma != otro.forma) throw std::invalid_argument("las formas de los tensores deben coincidir para la suma");
            Tensor resultado(forma);
            for (size_t i = 0; i < datos.size(); ++i) resultado.datos[i] = datos[i] + otro.datos[i];
            return resultado;}

        Tensor operator-(const Tensor& otro) const {
            if (forma != otro.forma) throw std::invalid_argument("las formas de los tensores deben coincidir para la resta");
            Tensor resultado(forma);
            for (size_t i = 0; i < datos.size(); ++i) resultado.datos[i] = datos[i] - otro.datos[i];
            return resultado;}

        Tensor operator*(const Tensor& otro) const {
            for (size_t i = 0; i < tam; ++i) {
                if (forma[i] != otro.forma[i] && forma[i] != 1 && otro.forma[i] != 1) {
                    throw std::invalid_argument("tamaños incompatibles para la multiplicación de tensores");}}
            std::array<size_t, tam> forma_resultado;
            for (size_t i = 0; i < tam; ++i)
                forma_resultado[i] = std::max(forma[i], otro.forma[i]);

            Tensor resultado{forma_resultado};
            std::array<size_t, tam> indices{};
            multiply_with_broadcasting(resultado, *this, otro, indices);
            return resultado;}

        Tensor operator*(const T& escalar) const {
            Tensor resultado(forma);
            for (size_t i = 0; i < datos.size(); ++i) resultado.datos[i] = datos[i] * escalar;
            return resultado;}

        Tensor transpose_2d() const {
            static_assert(tam == 2, "solo disponible para tensores 2d");
            std::array<size_t, 2> nueva_forma{forma[1], forma[0]};
            Tensor resultado(nueva_forma);
            for (size_t i = 0; i < forma[0]; ++i) {
                for (size_t j = 0; j < forma[1]; ++j) {
                    resultado(j, i) = (*this)(i, j);}}
            return resultado;}};}
