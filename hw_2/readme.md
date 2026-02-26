# Задача 2
## Часть 1. Аналитическое выражение для коэффициентов.

Для начальной волновой функции $\psi_0(x)$ можно рассмотреть эволюцию волновой функции в базисе $\{\phi_n(x) = \sqrt{\frac{2}{L}} \sin \left(\frac{\pi n x}{L} \right) \}$ решений бесконечной ямы ширины $L$. Так как этот базис диагонализирует гамильтониан, то

$$
\psi(x,t) = \sum_{n} c_n \phi_n(x) e^{-i E_n t}
$$

где 

$$
c_n = \int_{0}^{L} dx\ \phi_n^{*}(x) \psi_0(x)
$$ и 

$$
E_n = \frac{1}{2m}(\frac{\pi n}{L})^2
$$ 

∂В случае

$$
\psi_0(x) = \frac{1}{(2\pi \sigma^2)^{1/4}}\ \exp\left(-\frac{(x - x_0)^2}{4\sigma^2} \right)\ e^{ik_0x}
$$

Мы будем иметь 
$$
c_n = \sqrt{\frac{2}{L}} \frac{1}{(2\pi \sigma^2)^{1/4}}\int_{0}^{L} \sin (\frac{\pi n x}{L}) e^{-\frac{(x - x_0)^2}{4\sigma^2}} e^{ik_0 x} dx
$$

После плодотворных вычислений можно прийти к аналитическому ответу

$$
c_n = -i \sqrt{\frac{\sigma}{2L}} \left(\frac{\pi}{2} \right)^{1/4} \exp\left(-\frac{x_0^2}{4\sigma^2}\right) \left[e^{\frac{t_1^2}{4\sigma^2}} \left( \text{erf} \left(\frac{L - t_1}{2\sigma} \right)  + \text{erf} \left(\frac{t_1}{2\sigma} \right)\right) - e^{\frac{t_2^2}{4\sigma^2}} \left(\text{erf} \left(\frac{L - t_2}{2\sigma} \right)  + \text{erf} \left(\frac{t_2}{2\sigma}\right)\right)\right]  
$$
где $t_1 = 2x_0 + 4i\sigma^2 \left(\frac{\pi n}{L} - k_0 \right)$, $t_2 = 2x_0 - 4i\sigma^2 \left(\frac{\pi n}{L} + k_0 \right)$, а $\text{erf}(x) = \frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-\xi^2} d\xi$.

Однако данное решение плохо работает с данными с плавающей точкой, так как $t_1/2\sigma  \propto x_0/\sigma$, что в случае малых $\sigma$ улетает в бесконечность. Из-за этого будет страдать и функция ошибок Гаусса и множители перед ними. Можно переписать выражение в форме, устойчивой к таким ошибкам

$$
\begin{cases}
c_n = -i \sqrt{\frac{\sigma}{2L}} \left(\frac{\pi}{2} \right)^{1/4} \cdot \left[E(k_+) - E(k_{-}) \right]\\
E(k_{\pm}) = 2e^{-\sigma^2 k_{\pm}^{2} + i k_{\pm} x_0} - e^{-\frac{(L-x_0)^2}{4\sigma^2} + i k_{\pm} L}\cdot \omega(iz_1) - e^{-\frac{x_0^2}{4\sigma^2}} \omega(iz_2)\\
iz_1 = \sigma k_{\pm} + i \frac{L - x_0}{2\sigma},\quad iz_2 = -\sigma k_{\pm} + i\frac{x_0}{2\sigma}\\
k_{\pm} = k_0 \pm \frac{\pi n}{L}
\end{cases}
$$

где $\omega(iz_1)$ - [функция Фадеевой](https://ru.wikipedia.org/wiki/Функция_Фаддеевой) (позволяет использовать сходящуюся функцию в мнимой и комплексных частях). 

Теперь попробуем оценить $N$ - предельный индекс ряда. Во первых, оказывается, что оценку можно произвести только по первому слагаемому $E(k_{\pm})$, так как остальные малы (при большом $n$). Тогда все затухание заключено в $e^{-(\sigma k_{\pm})^2}$. Для примера оценку сделаем для $k_{-}$ и потребуем

$$
\sigma^2 (\frac{\pi n}{L} - k_0)^2 \geq M \Rightarrow n \geq \frac{L}{\pi} \left(k_0 + \frac{\sqrt{M}}{\sigma} \right)
$$

Пусть $\sqrt{M}=10$, тогда $N = \lceil \frac{L}{\pi} \left(k_0 + \frac{10}{\sigma} \right) \rceil$ - оценка числа слагаемых в ряде (число базисных функций). 


## Часть 2. Добавляем барьер.

На семинаре решение искалось в виде 

$$
\ket{\psi(t)} = \sum_{n = 1}^{N} c_n(t) \ket{\phi_n}
$$

а гамильтониан имел вид 
$$
\hat{H} = \frac{\hat{p}^2}{2m} + V(x)
$$

где 

$$
V(x) = \begin{cases}
0,\ \text{если}\ x \notin[x_b, x_b + w_b]\\
V_0,\ \text{если}\ x \in[x_b, x_b + w_b]
\end{cases}
$$

На семинаре пришли к решению для вектора амплитуд 

$$
\boldsymbol{c}(t) = \exp \left(-i Ht \right) \boldsymbol{c}(0)
$$

где матричный элемент гамильтониана в этом представлении

$$
H_{mn} = E_n \delta_{mn} + V_0 \underbrace{\int_{x_b}^{x_b + w_b}dx\ \phi^{*}_{m}(x) \phi_n(x)}_{I}
$$

Посчитаем этот интегральчик

$$
I = \frac{2}{L} \int_{x_b}^{x_b + w_b}dx\ \sin \left(\frac{\pi m x}{L} \right) \sin \left( \frac{\pi n x}{L}\right) = \frac{1}{\pi} \left[\frac{\sin \left(\frac{\pi x}{L} (m - n)\right)}{m-n} - \frac{\sin \left(\frac{\pi x}{L} (m + n)\right)}{m+n} \right] \Big|_{x_b}^{x_b + w_b}
$$

Итого

$$
I = \frac{1}{\pi} \left[\frac{\sin \left(\frac{\pi (x_b + w_b)}{L} (m - n)\right) - \sin \left(\frac{\pi x_b}{L} (m - n)\right)}{m-n} - \frac{\sin \left(\frac{\pi (x_b + w_b)}{L} (m + n)\right) - \sin \left(\frac{\pi x_b}{L} (m + n)\right)}{m+n}  \right]
$$

Но такая формула будем принята компьютером только в случае $m \neq n$. Иначе

$$
I_{m = n} = \frac{w_b}{L} - \frac{1}{2\pi} \frac{\sin \left(\frac{\pi (x_b + w_b)}{L} 2n \right) - \sin \left(\frac{2\pi x_b}{L} n\right)}{n}
$$

Таким образом имеем

$$
H_{mn} = \begin{cases}
E_n + V_0 \left[ \frac{w_b}{L} - \frac{1}{2\pi} \frac{\sin \left(\frac{\pi (x_b + w_b)}{L} 2n \right) - \sin \left(\frac{2\pi x_b}{L} n\right)}{n} \right],\ n =m \\
\frac{V_0}{\pi} \left[\frac{\sin \left(\frac{\pi (x_b + w_b)}{L} (m - n)\right) - \sin \left(\frac{\pi x_b}{L} (m - n)\right)}{m-n} - \frac{\sin \left(\frac{\pi (x_b + w_b)}{L} (m + n)\right) - \sin \left(\frac{\pi x_b}{L} (m + n)\right)}{m+n}  \right],\ n\neq m
\end{cases}
$$

Таким образом мы можем посчитать матрицу гамильтониана в этом базисе.

## Часть 3. Сравнение с методом Эйлера.
Возьмем параметры 

$$
    L = 6.0,\ x_0 = 1.0,\ \sigma = 0.3,\ m = 1.0,\ k_0 = 20,\ H = 100,\ x_b = 3,\ w_b = 1,\ N_{\text{сетка}} = 1000,\ T = 1
$$

Для них получается такое сравнение решений

![Comparision](euler_vs_series.gif)