# -*- coding: utf-8 -*-
"""
Costas-Sidon Code (CSC) - учебный прототип звукового матричного кода.
Нужен только numpy:  python -m pip install numpy

ВСЁ В ОДНОМ ФАЙЛЕ: запустите (F5 в IDLE или `python csc_all.py`) - и программа
сама выполнит все проверки, а в конце предложит проиграть звук и декодировать
вашу запись с телефона.

Что делает программа при запуске:
  1. Проверяет теоремы 1 и 2 перебором (массив Костаса, множество Сидона).
  2. Считает "призраки" третьего порядка для трёх сеток частот.
  3. Кодирует текст в WAV-файл (csc_message.wav) и декодирует его обратно.
  4. Пропускает сообщение через модель канала: нелинейный динамик,
     неровная АЧХ, шум.
  5. Сравнивает долю ошибок на символ для трёх сеток частот.
"""
import itertools
import wave
import numpy as np

# ---------------------------------------------------------------- параметры
FS = 44100          # частота дискретизации, Гц
N = 4096            # отсчётов в одном символе (~0,093 с)
DF = FS / N         # шаг по частоте, ~10,77 Гц
K = 3               # тонов в символе
BITS = 7            # бит на символ: floor(log2(165)) = 7
OFFSET = 93         # первый рабочий бин (~1001 Гц)

# оптимальная линейка Голомба из 11 меток (длина 72)
GOLOMB = [0, 1, 4, 13, 28, 33, 47, 54, 64, 70, 72]


# ------------------------------------------------- математика (теоремы 1-3)
def primitive_root(p):
    """Первообразный корень по модулю простого p (перебором)."""
    for g in range(2, p):
        if len({pow(g, i, p) for i in range(1, p)}) == p - 1:
            return g
    return 1


def welch_costas(p, g=None):
    """Построение Уэлча: f(i) = g^i mod p, i = 1..p-1."""
    g = g or primitive_root(p)
    return [pow(g, i, p) for i in range(1, p)]


def is_costas(perm):
    """Массив Костаса: перестановка 1..n, у каждого сдвига h разности различны."""
    n = len(perm)
    if sorted(perm) != list(range(1, n + 1)):
        return False
    for h in range(1, n):
        d = [perm[i + h] - perm[i] for i in range(n - h)]
        if len(d) != len(set(d)):
            return False
    return True


def is_sidon(a):
    """Множество Сидона: все суммы x+y (x<=y) различны."""
    s = [x + y for i, x in enumerate(a) for y in a[i:]]
    return len(s) == len(set(s))


def erdos_turan(p):
    """Построение Эрдёша-Турана: a_k = 2pk + (k^2 mod p)."""
    return [2 * p * k + (k * k) % p for k in range(p)]


def third_order_collisions(positions):
    """Сколько продуктов 2fi-fj и fi+fj-fk попадают на рабочую частоту."""
    P = list(positions)
    S = set(P)
    cnt = 0
    for i in P:
        for j in P:
            if i != j and (2 * i - j) in S:
                cnt += 1
    for i, j in itertools.combinations(P, 2):
        for k in P:
            if k != i and k != j and (i + j - k) in S:
                cnt += 1
    return cnt


# ----------------------------------------------------------- сетки частот
def grid_uniform():
    return [OFFSET + 37 * i for i in range(11)]


def grid_golomb():
    return [OFFSET + 5 * m for m in GOLOMB]


def grid_erdos_turan():
    return [OFFSET + x for x in erdos_turan(11)]


# ------------------------------------------------------------ звук и канал
COMBOS = list(itertools.combinations(range(11), K))          # 165 троек
COMBOS_INDEX = {c: s for s, c in enumerate(COMBOS)}          # тройка -> символ


def tone_frame(bins):
    """Один кадр: сумма синусоид на заданных бинах ДПФ (амплитуда <= 1)."""
    t = np.arange(N) / FS
    x = sum(np.sin(2 * np.pi * b * DF * t) for b in bins)
    return x / len(bins)


def encode_symbol(sym, positions):
    return tone_frame([positions[i] for i in COMBOS[sym]])


def nonlinear(x, a2, a3):
    """Модель нелинейного динамика: y = x + a2*x^2 + a3*x^3."""
    return x + a2 * x ** 2 + a3 * x ** 3


def random_response(x, span_db, rng):
    """Неровная АЧХ: гладкая случайная кривая с размахом span_db (дБ).
    Своя в каждом испытании - так имитируется разница между устройствами."""
    if span_db <= 0:
        return x
    spec = np.fft.rfft(x)
    n_nodes = len(spec) // 20 + 2          # неровности каждые ~20 бинов (~215 Гц)
    nodes = rng.normal(size=n_nodes)
    curve = np.interp(np.arange(len(spec)), np.linspace(0, len(spec) - 1, n_nodes), nodes)
    curve = (curve - curve.min()) / (curve.max() - curve.min() + 1e-12)   # 0..1
    gain = 10 ** ((curve * span_db - span_db / 2) / 20)
    return np.fft.irfft(spec * gain, n=len(x))


def add_noise(x, snr_db, rng):
    """Гауссов шум с заданным отношением сигнал/шум (дБ)."""
    p_sig = np.mean(x ** 2)
    p_noise = p_sig / (10 ** (snr_db / 10))
    return x + rng.normal(0, np.sqrt(p_noise), size=len(x))


def decode_symbol(frame, positions, window=None):
    """Три самых сильных рабочих бина -> номер символа (или -1)."""
    if window is not None:
        frame = frame * window
    mags = np.abs(np.fft.rfft(frame)[positions])
    top = tuple(sorted(int(i) for i in np.argsort(mags)[-K:]))
    return COMBOS_INDEX.get(top, -1)


# ------------------------------------------- сообщение: биты, CRC, символы
def crc8(data):
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def message_to_symbols(text):
    """Текст -> [байт длины] + байты + CRC-8 -> группы по 7 бит -> символы."""
    data = text.encode("utf-8")
    if len(data) > 255:
        raise ValueError("Сообщение длиннее 255 байт")
    payload = bytes([len(data)]) + data
    payload += bytes([crc8(payload)])
    bits = "".join(f"{b:08b}" for b in payload)
    bits += "0" * (-len(bits) % BITS)
    return [int(bits[i:i + BITS], 2) for i in range(0, len(bits), BITS)]


def symbols_to_message(symbols):
    """Обратное преобразование; возвращает (текст, crc_ok)."""
    if any(s < 0 or s >= 2 ** BITS for s in symbols):
        return None, False
    bits = "".join(f"{s:07b}" for s in symbols)
    raw = bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits) - 7, 8))
    if not raw:
        return None, False
    n = raw[0]
    if len(raw) < n + 2:
        return None, False
    payload, crc = raw[:n + 1], raw[n + 1]
    ok = crc8(payload) == crc
    try:
        return payload[1:].decode("utf-8"), ok
    except UnicodeDecodeError:
        return None, False


# ---------------------------------- преамбула Костаса и поиск начала
PREAMBLE = welch_costas(11)            # 10 символов, значения 1..10


def build_signal(text, positions, pause=0.2):
    """Пауза + преамбула Костаса + данные + пауза."""
    gap = np.zeros(int(pause * FS))
    frames = [tone_frame([positions[f - 1]]) for f in PREAMBLE]
    frames += [encode_symbol(s, positions) for s in message_to_symbols(text)]
    return np.concatenate([gap] + frames + [gap])


def find_start(sig, positions, step=128):
    """Ищем сдвиг, при котором преамбула Костаса совпадает лучше всего."""
    def score(off):
        total = 0.0
        for i, f in enumerate(PREAMBLE):
            seg = sig[off + i * N: off + (i + 1) * N]
            if len(seg) < N:
                return -1
            m = np.abs(np.fft.rfft(seg)[positions])
            total += m[f - 1] / (m.sum() + 1e-12)
        return total

    last = len(sig) - len(PREAMBLE) * N
    best = max(range(0, max(last, 1), step), key=score)
    lo, hi = max(0, best - step), min(last, best + step)
    return max(range(lo, hi + 1, 8), key=score)


def decode_signal(sig, positions, window=None):
    start = find_start(sig, positions)
    pos = start + len(PREAMBLE) * N

    def sym(i):
        return decode_symbol(sig[pos + i * N: pos + (i + 1) * N], positions, window)

    first = [sym(0), sym(1)]                     # 14 бит - хватит на байт длины
    if -1 in first:
        return None, False
    n = int(("".join(f"{s:07b}" for s in first))[:8], 2)
    n_sym = -(-8 * (n + 2) // BITS)              # потолок деления
    symbols = [sym(i) for i in range(n_sym)]
    return symbols_to_message(symbols)


def save_wav(sig, filename):
    data = (np.clip(sig, -1, 1) * 32767 * 0.8).astype("<i2")
    with wave.open(filename, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(FS)
        w.writeframes(data.tobytes())


def load_wav(filename):
    with wave.open(filename, "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(float) / (32767 * 0.8)


# --------------------------------------------------------- эксперимент
def symbol_error_rate(positions, a3, snr_db=20, n_symbols=1000, a2=0.2,
                      response_db=20, seed=1):
    """Доля неверно принятых символов в модели канала."""
    rng = np.random.default_rng(seed)
    errors = 0
    for _ in range(n_symbols):
        s = int(rng.integers(0, 2 ** BITS))
        y = nonlinear(encode_symbol(s, positions), a2, a3)
        y = random_response(y, response_db, rng)
        y = add_noise(y, snr_db, rng)
        if decode_symbol(y, positions) != s:
            errors += 1
    return errors / n_symbols


# ================================================================= проверки
def ok(flag):
    return "OK " if flag else "ОШИБКА"


def run_checks(grids):
    """Автотесты: каждая строка - ОК или ОШИБКА, в конце общий итог."""
    results = []

    def check(name, flag, extra=""):
        results.append(flag)
        print(f"  [{ok(flag)}] {name} {extra}")

    print("=== 1. Проверка теорем перебором ===")
    f = welch_costas(11)
    check("Уэлч p=11 - массив Костаса", is_costas(f), str(f))
    check("Развёртка 1..10 НЕ массив Костаса", not is_costas(list(range(1, 11))))
    a = erdos_turan(5)
    check("Эрдёш-Туран p=5 - множество Сидона", is_sidon(a), str(a))
    check("Голомб (11 меток) - множество Сидона", is_sidon(GOLOMB))
    check("Равномерная сетка НЕ множество Сидона", not is_sidon(grid_uniform()))

    print("\n=== 2. Призраки третьего порядка (теорема 3) ===")
    counts = {n: third_order_collisions(p) for n, p in grids.items()}
    for n, c in counts.items():
        print(f"      {n:12s}: {c} совпадений")
    check("Равномерная сетка: 355 совпадений", counts["равномерная"] == 355)
    check("Эрдёш-Туран: 0 совпадений", counts["Эрдёш-Туран"] == 0)
    check("Голомб: 0 совпадений", counts["Голомб"] == 0)

    print("\n=== 3. Кодирование и декодирование (без искажений) ===")
    pos = grid_golomb()
    for text in ["Hello", "Zvuk QR 2026", "Привет, мир!", "a" * 100]:
        sig = build_signal(text, pos)
        got, crc_ok = decode_signal(sig, pos)
        check(f"текст {text[:14]!r}", got == text and crc_ok)
    text = "Zvuk QR 2026"
    sig = build_signal(text, pos)
    save_wav(sig, "csc_message.wav")
    got, crc_ok = decode_signal(load_wav("csc_message.wav"), pos)
    check("через WAV-файл csc_message.wav (сохранён рядом с программой)",
          got == text and crc_ok, f"{len(sig) / FS:.2f} с")

    print("\n=== 4. Программа должна ломаться там, где положено ===")
    syms = message_to_symbols(text)
    syms[5] = (syms[5] + 1) % 128
    got, crc_ok = symbols_to_message(syms)
    check("испорченный символ -> CRC даёт ошибку", not crc_ok, f"(получено {got!r})")
    shifted = np.concatenate([np.zeros(12345), sig])
    got, crc_ok = decode_signal(shifted, pos)
    check("начало сообщения найдено по преамбуле Костаса", got == text and crc_ok)
    rng = np.random.default_rng(1)
    for snr in (20, 0, -10):
        got, crc_ok = decode_signal(add_noise(sig, snr, rng), pos)
        check(f"шум, ОСШ {snr} дБ", got == text and crc_ok)
    got, crc_ok = decode_signal(add_noise(sig, -20, rng), pos)
    check("ОСШ -20 дБ: декодер честно сдаётся (CRC не сходится)", not crc_ok)

    print("\n=== 5. Нелинейный динамик (a2=0.2, a3=1.0, АЧХ 20 дБ, ОСШ 20 дБ) ===")
    rng = np.random.default_rng(5)
    for name, p in grids.items():
        s = build_signal(text, p)
        y = add_noise(random_response(nonlinear(s, 0.2, 1.0), 20, rng), 20, rng)
        got, crc_ok = decode_signal(y, p)
        check(f"{name}: сообщение принято", got == text and crc_ok)

    print(f"\nИТОГ: {sum(results)} из {len(results)} проверок пройдено.")
    return all(results)


def run_experiment(grids):
    print("\n=== 6. Доля ошибочных символов (1000 символов на точку) ===")
    print("(размах АЧХ 20 дБ; это учебная модель, числа могут отличаться от таблиц отчёта)")
    print("Это займёт около минуты...")
    for a3 in (0.0, 0.5, 1.0, 2.0):
        row = ", ".join(f"{n} {100 * symbol_error_rate(p, a3):5.1f}%"
                        for n, p in grids.items())
        print(f"a3 = {a3}: {row}")


def draw_spectrogram(filename="csc_message.wav", out="csc_spectrogram.png"):
    """Картинка 'звукового QR-кода' (нужен matplotlib; если его нет - пропуск)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(Спектрограмма пропущена: для неё нужен matplotlib - "
              "python -m pip install matplotlib)")
        return
    sig = load_wav(filename)
    fig, ax = plt.subplots(figsize=(10, 4))
    with np.errstate(divide="ignore"):      # в паузах - тишина, log(0) безвреден
        ax.specgram(sig, NFFT=N, Fs=FS, noverlap=0, cmap="magma")
    ax.set_ylim(0, 6000)
    ax.set_xlabel("время, с")
    ax.set_ylabel("частота, Гц")
    ax.set_title("Спектрограмма сообщения CSC: преамбула Костаса и данные")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"\nСпектрограмма сохранена: {out}")


def play_wav(filename="csc_message.wav"):
    """Проигрывание (только Windows, встроенный winsound)."""
    try:
        import winsound
        winsound.PlaySound(filename, winsound.SND_FILENAME)
    except ImportError:
        print("Проигрывание из программы работает только в Windows - "
              "откройте csc_message.wav в любом плеере.")


def decode_recording(path, grids):
    """Декодирует запись (WAV, 44100 Гц, моно, 16 бит) на всех трёх сетках."""
    try:
        with wave.open(path, "rb") as w:
            fs, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
    except (FileNotFoundError, wave.Error) as e:
        print("Не удалось открыть файл:", e)
        return
    if (fs, ch, width) != (FS, 1, 2):
        print(f"Нужен WAV 44100 Гц, моно, 16 бит; у файла {fs} Гц, "
              f"{ch} кан., {8 * width} бит. Сконвертируйте в Audacity.")
        return
    sig = load_wav(path)
    for name, p in grids.items():
        try:
            got, crc_ok = decode_signal(sig, p)
        except Exception as e:           # слишком короткая или пустая запись
            got, crc_ok = None, False
        print(f"  {name:12s}: {got!r}  CRC {'сошлась' if crc_ok else 'НЕ сошлась'}")
    print("(Запись должна быть сделана с сетки Голомба - это сетка в файле csc_message.wav.)")


if __name__ == "__main__":
    grids = {
        "равномерная": grid_uniform(),
        "Эрдёш-Туран": grid_erdos_turan(),
        "Голомб": grid_golomb(),
    }
    run_checks(grids)
    run_experiment(grids)
    draw_spectrogram()

    print("\n=== Дополнительно ===")
    if input("Проиграть csc_message.wav? (y/n): ").strip().lower() == "y":
        play_wav()
    path = input("Путь к записи с телефона (WAV) или Enter, чтобы пропустить: ").strip().strip('"')
    if path:
        decode_recording(path, grids)
    print("\nГотово.")
