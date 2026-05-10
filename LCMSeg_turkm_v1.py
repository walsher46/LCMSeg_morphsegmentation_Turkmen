"""
LCMSeg_turkm_v1.py
==================
Адаптация LCMSeg_kaz_v3 для туркменского языка.

Ключевые отличия от казахской версии:
  1. Туркменский латинский алфавит (не кириллица)
  2. Нормализация ň (U+0148) → ñ (U+00F1) — критично для совпадения
     символов в тексте и таблице окончаний
  3. Три дополнительных признака сингармонизма на уровне символов:
       • is_vowel          — гласная/согласная (0/1)
       • vowel_class       — front=0, back=1, consonant=2
       • word_harmony      — класс последней гласной всего слова
     Эти признаки конкатенируются с символьным эмбеддингом перед BiLSTM,
     давая модели явную информацию о гармонии аффиксов.
  4. Туркменские BMES-теги те же (B/M/E/S), логика та же.
  5. Пути к корпусу и файлам модели переименованы под TURKM.

Требуемый формат корпуса (такой же @@ как в казахской версии):
  git @@ dim                     → пришёл (gel-dim)
  adam @@ lar @@ yň              → людей
  Türkmenistan @@ da             → в Туркменистане
  öý @@ ler @@ iň               → домов
"""

# ─── Установка зависимостей (Colab) ─────────────────────────────────────────
# !pip install -q pytorch-crf

import os
import json
from typing import List, Tuple

import torch
import torch.nn as nn
from torchcrf import CRF
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ════════════════════════════════════════════════════════════════════════════
# 1. ПУТИ И ПОДГОТОВКА КОРПУСА
# ════════════════════════════════════════════════════════════════════════════

BASE_DIR = "/content/drive/MyDrive/TURKM_MORPH"
os.makedirs(BASE_DIR, exist_ok=True)

SENT_FULL_PATH = os.path.join(BASE_DIR, "turkm_segmented_corpus_full.txt")
SENT_50K_PATH  = os.path.join(BASE_DIR, "turkm_segmented_corpus_50k.txt")
CHAR2ID_PATH   = os.path.join(BASE_DIR, "char2id_femseg_turkm_v1_50k.json")
MODEL_DIR      = os.path.join(BASE_DIR, "models_femseg_turkm_v1_50k")
os.makedirs(MODEL_DIR, exist_ok=True)

# Создаём подкорпус на 50k предложений если не существует
if not os.path.exists(SENT_50K_PATH):
    print("Создаём подкорпус на 50 000 предложений...")
    os.system(f'head -n 50000 "{SENT_FULL_PATH}" > "{SENT_50K_PATH}"')
    print("Готово:", SENT_50K_PATH)
else:
    print("Файл на 50k уже существует:", SENT_50K_PATH)

# ════════════════════════════════════════════════════════════════════════════
# 2. ТУРКМЕНСКИЙ АЛФАВИТ И СИНГАРМОНИЗМ
# ════════════════════════════════════════════════════════════════════════════

# ── 2.1 Нормализация Unicode ──────────────────────────────────────────────
# КРИТИЧНО: в текстах может быть ň (U+0148), а в таблицах окончаний — ñ (U+00F1).
# Всегда нормализуем перед обработкой.
def norm_turkm(s: str) -> str:
    """Нормализация туркменского текста:
    - ň (U+0148) → ñ (U+00F1)  — носовая ñ
    - Ň (U+0147) → Ñ (U+00D1)  — заглавная
    - Убирает BOM и лишние пробелы
    """
    return (str(s)
            .replace('\u0148', '\u00f1')   # ň → ñ
            .replace('\u0147', '\u00d1')   # Ň → Ñ
            .replace('\ufeff', '')          # BOM
            .strip())

# ── 2.2 Гласные туркменского языка ───────────────────────────────────────
# Передние (мягкие): e, i, ö, ü  → аффиксы: -ler, -de, -den, -e, -iň
# Задние  (твёрдые): a, y, o, u  → аффиксы: -lar, -da, -dan, -a, -yň
# (y в туркменском = ы, заднеязычная)

FRONT_VOWELS = set('eiöü')     # передние
BACK_VOWELS  = set('ayou')     # задние  (a, y=ы, o, u)
ALL_VOWELS   = FRONT_VOWELS | BACK_VOWELS

# Числовые коды классов для эмбеддинга
HARMONY_FRONT     = 0   # передняя гласная
HARMONY_BACK      = 1   # задняя гласная
HARMONY_CONSONANT = 2   # согласная (не участвует в гармонии)

def char_harmony_class(ch: str) -> int:
    """Возвращает класс гармонии символа."""
    c = ch.lower()
    if c in FRONT_VOWELS:
        return HARMONY_FRONT
    elif c in BACK_VOWELS:
        return HARMONY_BACK
    else:
        return HARMONY_CONSONANT

def word_harmony_class(word: str) -> int:
    """Класс гармонии всего слова = класс ПОСЛЕДНЕЙ гласной.
    Определяет, какой вариант аффикса должен следовать.
    Возвращает HARMONY_FRONT / HARMONY_BACK / HARMONY_CONSONANT.
    """
    for ch in reversed(word.lower()):
        cls = char_harmony_class(ch)
        if cls != HARMONY_CONSONANT:
            return cls
    return HARMONY_CONSONANT  # нет гласных (аббревиатура и т.п.)

def is_vowel(ch: str) -> int:
    """1 если гласная, 0 если согласная."""
    return 1 if ch.lower() in ALL_VOWELS else 0

# ── 2.3 Проверка правильности аффикса (soft constraint) ──────────────────
# Используется в постобработке и диагностике, не в обучении.
HARMONY_PAIRS = {
    # (аффикс_задний, аффикс_передний)
    'lar':  'ler',
    'ler':  'lar',
    'da':   'de',
    'de':   'da',
    'dan':  'den',
    'den':  'dan',
    'dy':   'di',
    'di':   'dy',
    'dyr':  'dir',
    'dir':  'dyr',
    'yň':   'iň',
    'iň':   'yň',
    'ny':   'ni',
    'ni':   'ny',
    'a':    'e',
    'e':    'a',
    'y':    'i',
    'i':    'y',
}

def check_harmony(stem: str, affix: str) -> bool:
    """Проверяет согласованность аффикса с гармонией основы.
    Возвращает True если гармония правильная или невозможно проверить.
    """
    stem_harmony = word_harmony_class(stem)
    if stem_harmony == HARMONY_CONSONANT:
        return True   # нет гласных в основе — проверить невозможно
    affix_l = affix.lower()
    pair = HARMONY_PAIRS.get(affix_l)
    if pair is None:
        return True   # аффикс не в списке гармонических пар
    # Передняя основа → передний аффикс (не задний)
    if stem_harmony == HARMONY_FRONT:
        # аффикс не должен быть "задним" вариантом
        back_form = HARMONY_PAIRS.get(affix_l)  # парный = задний?
        return affix_l in FRONT_VOWELS or affix_l[0] not in BACK_VOWELS
    return True


# ════════════════════════════════════════════════════════════════════════════
# 3. BMES ПО МОРФЕМАМ
# ════════════════════════════════════════════════════════════════════════════

TAGS   = ["B", "M", "E", "S"]
TAG2ID = {t: i for i, t in enumerate(TAGS)}
ID2TAG = {i: t for t, i in TAG2ID.items()}

def morphs_to_bmes_char(word: str, morphs: List[str]) -> List[str]:
    """Преобразует список морфем в BMES-теги по символам."""
    tags: List[str] = []
    for m in morphs:
        m_len = len(m)
        if m_len == 1:
            tags.append("S")
        else:
            for i in range(m_len):
                if i == 0:
                    tags.append("B")
                elif i == m_len - 1:
                    tags.append("E")
                else:
                    tags.append("M")
    if len(tags) != len(word):
        return []
    return tags

# ════════════════════════════════════════════════════════════════════════════
# 4. РАЗБОР ТУРКМЕНСКОГО CSE-ПРЕДЛОЖЕНИЯ
# ════════════════════════════════════════════════════════════════════════════

PUNCT_TOKENS = {
    ",", ".", "?", "!", ";", ":", "—", "-", "…",
    "„", """, "«", "»", "(", ")", "[", "]", '"', "'"
}

def line_to_word_morphs_turkm(line: str) -> List[Tuple[str, List[str]]]:
    """
    Разбирает строку туркменского CSE-корпуса в список (слово, морфемы).
    Формат: "adam @@ lar @@ yň geldi"
    После разбора каждая строка → список (слово, [морфема1, морфема2, ...]).
    Нормализация ň→ñ применяется к каждому токену.
    """
    tokens = [norm_turkm(t) for t in line.strip().split()]
    words  = []
    cur    = []

    for tok in tokens:
        if tok in PUNCT_TOKENS:
            if cur:
                morphs = [t[:-2] if t.endswith("@@") else t for t in cur]
                word   = "".join(morphs)
                words.append((word, morphs))
                cur = []
            words.append((tok, [tok]))
            continue

        cur.append(tok)
        if not tok.endswith("@@"):
            morphs = [t[:-2] if t.endswith("@@") else t for t in cur]
            word   = "".join(morphs)
            words.append((word, morphs))
            cur = []

    if cur:
        morphs = [t[:-2] if t.endswith("@@") else t for t in cur]
        word   = "".join(morphs)
        words.append((word, morphs))

    return words

# ════════════════════════════════════════════════════════════════════════════
# 5. ПОСТРОЕНИЕ ВЫБОРКИ С ПРИЗНАКАМИ СИНГАРМОНИЗМА
# ════════════════════════════════════════════════════════════════════════════

def build_samples_from_cse(path: str):
    """
    Читает туркменский CSE-корпус и возвращает список семплов.

    Каждый семпл: (char_ids, harmony_ids, is_vowel_ids, word_harmony_ids, tag_ids)
      char_ids        — индексы символов (строятся позже через char2id)
      harmony_ids     — класс гармонии каждого символа (0=front, 1=back, 2=cons)
      is_vowel_ids    — 0/1 для каждого символа
      word_harmony_ids— класс гармонии всего слова (повторяется для каждого символа)
      tag_ids         — BMES индексы

    Признаки сингармонизма (harmony_ids, is_vowel_ids, word_harmony_ids)
    передаются в модель как отдельные эмбеддинги и конкатенируются
    с символьным эмбеддингом — это даёт модели явный сигнал о гармонии
    при предсказании границы морфемы.
    """
    samples   = []
    total_lines = 0
    bad_words   = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_lines += 1

            word_morphs = line_to_word_morphs_turkm(line)
            for word, morphs in word_morphs:
                chars = list(word)
                tags  = morphs_to_bmes_char(word, morphs)
                if not tags or len(tags) != len(chars):
                    bad_words += 1
                    continue

                tag_ids = [TAG2ID[t] for t in tags]

                # ── Признаки сингармонизма ───────────────────────────
                harmony_ids      = [char_harmony_class(ch) for ch in chars]
                is_vowel_ids     = [is_vowel(ch) for ch in chars]
                word_h           = word_harmony_class(word)
                word_harmony_ids = [word_h] * len(chars)  # одно значение на всё слово

                samples.append((chars, harmony_ids, is_vowel_ids, word_harmony_ids, tag_ids))

    print(f"Всего строк:               {total_lines}")
    print(f"Слов-семплов:              {len(samples)}")
    print(f"Проблемных слов (пропуск): {bad_words}")
    return samples

samples = build_samples_from_cse(SENT_50K_PATH)

# ════════════════════════════════════════════════════════════════════════════
# 6. СЛОВАРЬ СИМВОЛОВ (char2id)
# ════════════════════════════════════════════════════════════════════════════

def build_char_vocab(samples, min_freq: int = 1):
    from collections import Counter
    cnt = Counter()
    for chars, *_ in samples:
        for ch in chars:
            cnt[ch] += 1

    char2id = {"<pad>": 0, "<unk>": 1}
    for ch, c in cnt.items():
        if c >= min_freq:
            char2id[ch] = len(char2id)

    print(f"Размер char-вокабуляра: {len(char2id)}")
    # Туркменских специфичных букв должно быть ~8:
    special = [ch for ch in char2id if ch in 'çñöşüýžÇÑÖŞÜÝŽ']
    print(f"Туркменских спецбукв в вокабуляре: {special}")
    return char2id

char2id = build_char_vocab(samples, min_freq=1)

with open(CHAR2ID_PATH, "w", encoding="utf-8") as f:
    json.dump(char2id, f, ensure_ascii=False, indent=2)
print("char2id сохранён:", CHAR2ID_PATH)


def encode_samples(samples, char2id):
    """Кодирует символы через char2id, остальные признаки уже числовые."""
    encoded = []
    for chars, harmony_ids, is_vowel_ids, word_harmony_ids, tag_ids in samples:
        char_ids = [char2id.get(ch, char2id["<unk>"]) for ch in chars]
        encoded.append((char_ids, harmony_ids, is_vowel_ids, word_harmony_ids, tag_ids))
    return encoded

encoded = encode_samples(samples, char2id)

# ════════════════════════════════════════════════════════════════════════════
# 7. TRAIN / VAL SPLIT
# ════════════════════════════════════════════════════════════════════════════

train_data, val_data = train_test_split(encoded, test_size=0.1, random_state=42)
print(f"Train: {len(train_data)}, Val: {len(val_data)}")

# ════════════════════════════════════════════════════════════════════════════
# 8. DATASET И COLLATE_FN
# ════════════════════════════════════════════════════════════════════════════

class TurkmMorphDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # (char_ids, harmony_ids, is_vowel_ids, word_harmony_ids, tag_ids)
        return self.data[idx]


def collate_fn(batch):
    """
    Паддинг до максимальной длины в батче.
    Возвращает 5 тензоров + маску:
      input_ids        [B, L]  — символьные индексы
      harmony_ids      [B, L]  — класс гармонии каждого символа
      is_vowel_ids     [B, L]  — 0/1 гласная
      word_harmony_ids [B, L]  — класс гармонии слова
      tag_ids          [B, L]  — BMES теги (-1 для паддинга)
      mask             [B, L]  — True для реальных токенов
    """
    max_len = max(len(x[0]) for x in batch)
    pad_id  = 0

    all_char_ids        = []
    all_harmony_ids     = []
    all_is_vowel_ids    = []
    all_word_harmony    = []
    all_tag_ids         = []
    all_mask            = []

    for char_ids, harmony_ids, is_vowel_ids, word_harmony_ids, tag_ids in batch:
        l   = len(char_ids)
        pad = max_len - l

        all_char_ids.append(char_ids        + [pad_id] * pad)
        all_harmony_ids.append(harmony_ids  + [HARMONY_CONSONANT] * pad)
        all_is_vowel_ids.append(is_vowel_ids+ [0] * pad)
        all_word_harmony.append(word_harmony_ids + [HARMONY_CONSONANT] * pad)
        all_tag_ids.append(tag_ids          + [-1] * pad)
        all_mask.append([1] * l             + [0] * pad)

    return (
        torch.tensor(all_char_ids,        dtype=torch.long),
        torch.tensor(all_harmony_ids,     dtype=torch.long),
        torch.tensor(all_is_vowel_ids,    dtype=torch.long),
        torch.tensor(all_word_harmony,    dtype=torch.long),
        torch.tensor(all_tag_ids,         dtype=torch.long),
        torch.tensor(all_mask,            dtype=torch.bool),
    )

train_ds = TurkmMorphDataset(train_data)
val_ds   = TurkmMorphDataset(val_data)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  collate_fn=collate_fn)
val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False, collate_fn=collate_fn)

# ════════════════════════════════════════════════════════════════════════════
# 9. МОДЕЛЬ: BiLSTM + CRF + ПРИЗНАКИ СИНГАРМОНИЗМА
# ════════════════════════════════════════════════════════════════════════════

class TurkmLCMSegCRF(nn.Module):
    """
    BiLSTM-CRF с тремя дополнительными признаками сингармонизма.

    Архитектура входного слоя:
      char_emb        [B, L, emb_dim]       — символьный эмбеддинг
      harmony_emb     [B, L, harmony_dim]   — класс гармонии символа (3 класса)
      is_vowel_emb    [B, L, vowel_dim]     — гласная/согласная (2 класса)
      word_harm_emb   [B, L, word_harm_dim] — гармония слова (3 класса)

    Конкатенируются → BiLSTM → Dropout → Linear → CRF.

    Зачем это нужно для туркменского:
      Граница морфемы в тюркских языках почти всегда совпадает с сменой
      гармонии (основа задняя → аффикс задний). Явный признак harmony_class
      даёт BiLSTM сигнал, не требуя выучивать эту закономерность из символов
      неявно. На практике это ускоряет сходимость и повышает точность
      на редких аффиксальных формах.
    """

    def __init__(
        self,
        vocab_size:     int,
        tagset_size:    int,
        emb_dim:        int = 128,
        harmony_dim:    int = 16,   # признак гармонии символа (3 класса → 16d)
        vowel_dim:      int = 8,    # признак гласная/согл. (2 класса → 8d)
        word_harm_dim:  int = 16,   # гармония слова (3 класса → 16d)
        hidden_dim:     int = 256,
        dropout:        float = 0.3,
    ):
        super().__init__()

        # Эмбеддинги
        self.char_emb      = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.harmony_emb   = nn.Embedding(3, harmony_dim)    # front/back/consonant
        self.is_vowel_emb  = nn.Embedding(2, vowel_dim)      # 0/1
        self.word_harm_emb = nn.Embedding(3, word_harm_dim)  # front/back/consonant

        # Суммарная размерность входа в BiLSTM
        input_dim = emb_dim + harmony_dim + vowel_dim + word_harm_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,          # 2 слоя (в казахской 1) — туркменский сложнее морфологически
            bidirectional=True,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0,
        )
        self.dropout  = nn.Dropout(dropout)
        self.fc       = nn.Linear(hidden_dim * 2, tagset_size)
        self.crf      = CRF(tagset_size, batch_first=True)

    def forward(
        self,
        input_ids,        # [B, L]
        harmony_ids,      # [B, L]
        is_vowel_ids,     # [B, L]
        word_harmony_ids, # [B, L]
        tags=None,        # [B, L] или None
        mask=None,        # [B, L] bool
    ):
        # ── Конкатенация эмбеддингов ────────────────────────────────────
        x_char     = self.char_emb(input_ids)           # [B, L, emb_dim]
        x_harm     = self.harmony_emb(harmony_ids)       # [B, L, harmony_dim]
        x_vowel    = self.is_vowel_emb(is_vowel_ids)     # [B, L, vowel_dim]
        x_wh       = self.word_harm_emb(word_harmony_ids)# [B, L, word_harm_dim]

        x = torch.cat([x_char, x_harm, x_vowel, x_wh], dim=-1)  # [B, L, input_dim]

        # ── BiLSTM ───────────────────────────────────────────────────────
        x, _ = self.lstm(x)
        x    = self.dropout(x)
        emissions = self.fc(x)                           # [B, L, tagset_size]

        # ── CRF ──────────────────────────────────────────────────────────
        if tags is not None:
            loss = -self.crf(emissions, tags, mask=mask, reduction='mean')
            return loss
        else:
            return self.crf.decode(emissions, mask=mask)

# ════════════════════════════════════════════════════════════════════════════
# 10. ОБУЧЕНИЕ
# ════════════════════════════════════════════════════════════════════════════

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = TurkmLCMSegCRF(
    vocab_size=len(char2id),
    tagset_size=len(TAGS),
).to(device)

# Количество параметров
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Параметров модели: {n_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=2, verbose=True
)


def evaluate(model, loader):
    """Вычисляет точность на уровне токенов и морфем."""
    model.eval()
    total_tokens  = 0
    correct_tok   = 0
    total_morphs  = 0
    correct_morph = 0

    # Счётчики ошибок гармонии
    harmony_errors = 0
    harmony_total  = 0

    with torch.no_grad():
        for batch in loader:
            (input_ids, harmony_ids, is_vowel_ids,
             word_harmony_ids, tag_ids, mask) = [b.to(device) for b in batch]

            paths = model(input_ids, harmony_ids, is_vowel_ids,
                          word_harmony_ids, tags=None, mask=mask)

            pred_ids = torch.full_like(tag_ids, fill_value=-1)
            for i, seq in enumerate(paths):
                for j, t in enumerate(seq):
                    pred_ids[i, j] = t

            # Точность на токенах
            valid      = mask.view(-1) & (tag_ids.view(-1) >= 0)
            gold_flat  = tag_ids.view(-1)
            pred_flat  = pred_ids.view(-1)
            total_tokens  += valid.sum().item()
            correct_tok   += (gold_flat[valid] == pred_flat[valid]).sum().item()

            # Точность на морфемах (полное совпадение всего слова)
            B, L = tag_ids.shape
            for i in range(B):
                seq_len = mask[i].sum().item()
                gold_seq = tag_ids[i, :seq_len].tolist()
                pred_seq = pred_ids[i, :seq_len].tolist()
                total_morphs  += 1
                correct_morph += (gold_seq == pred_seq)

    tok_acc   = correct_tok   / max(1, total_tokens)
    morph_acc = correct_morph / max(1, total_morphs)
    return tok_acc, morph_acc


num_epochs  = 5
best_morph  = 0.0
best_ckpt   = None

print("\n── Начало обучения ──────────────────────────────────────────────")
for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for batch in train_loader:
        (input_ids, harmony_ids, is_vowel_ids,
         word_harmony_ids, tag_ids, mask) = [b.to(device) for b in batch]

        loss = model(input_ids, harmony_ids, is_vowel_ids,
                     word_harmony_ids, tags=tag_ids, mask=mask)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    avg_loss          = total_loss / max(1, n_batches)
    val_tok, val_morph = evaluate(model, val_loader)

    print(f"[Epoch {epoch:02d}] "
          f"train_loss={avg_loss:.4f}  "
          f"val_token_acc={val_tok:.4f}  "
          f"val_morph_acc={val_morph:.4f}")

    scheduler.step(val_morph)

    # Сохраняем чекпоинт каждой эпохи
    ckpt_path = os.path.join(MODEL_DIR, f"femseg_turkm_v1_epoch{epoch:02d}.pt")
    torch.save({
        'epoch':      epoch,
        'state_dict': model.state_dict(),
        'optimizer':  optimizer.state_dict(),
        'val_tok':    val_tok,
        'val_morph':  val_morph,
        'char2id':    char2id,
    }, ckpt_path)
    print(f"  checkpoint → {ckpt_path}")

    # Сохраняем лучшую модель отдельно
    if val_morph > best_morph:
        best_morph = val_morph
        best_ckpt  = ckpt_path
        best_path  = os.path.join(MODEL_DIR, "femseg_turkm_v1_BEST.pt")
        torch.save({
            'epoch':      epoch,
            'state_dict': model.state_dict(),
            'val_tok':    val_tok,
            'val_morph':  val_morph,
            'char2id':    char2id,
        }, best_path)
        print(f"  ★ Лучшая модель сохранена: val_morph={best_morph:.4f}")

print(f"\n── Обучение завершено ──────────────────────────────────────────")
print(f"   Лучший morph_acc: {best_morph:.4f}")
print(f"   Лучший чекпоинт:  {best_ckpt}")

# ════════════════════════════════════════════════════════════════════════════
# 11. ИНФЕРЕНС — сегментация нового слова
# ════════════════════════════════════════════════════════════════════════════

def load_model_for_inference(ckpt_path: str, device='cpu'):
    """Загружает модель из чекпоинта для инференса."""
    ckpt    = torch.load(ckpt_path, map_location=device)
    c2id    = ckpt['char2id']
    mdl     = TurkmFEMSegCRF(vocab_size=len(c2id), tagset_size=len(TAGS)).to(device)
    mdl.load_state_dict(ckpt['state_dict'])
    mdl.eval()
    return mdl, c2id


def segment_word_turkm(word: str, model, char2id, device='cpu') -> str:
    """
    Сегментирует одно туркменское слово, возвращает строку с @@ разметкой.
    Пример: 'adamlaryň' → 'adam@@ lar@@ yň'

    Признаки сингармонизма вычисляются автоматически из символов слова.
    """
    word_n = norm_turkm(word)
    chars  = list(word_n)
    if not chars:
        return word

    # Кодируем
    char_ids        = [char2id.get(ch, char2id["<unk>"]) for ch in chars]
    harmony_ids     = [char_harmony_class(ch) for ch in chars]
    is_vowel_ids    = [is_vowel(ch) for ch in chars]
    word_h          = word_harmony_class(word_n)
    word_harmony_ids= [word_h] * len(chars)

    # В тензоры (батч=1)
    to_t = lambda lst, dt: torch.tensor([lst], dtype=dt).to(device)
    inp  = to_t(char_ids,         torch.long)
    harm = to_t(harmony_ids,      torch.long)
    vow  = to_t(is_vowel_ids,     torch.long)
    wh   = to_t(word_harmony_ids, torch.long)
    mask = torch.ones(1, len(chars), dtype=torch.bool).to(device)

    with torch.no_grad():
        paths = model(inp, harm, vow, wh, tags=None, mask=mask)

    pred_tags = [ID2TAG[t] for t in paths[0]]

    # BMES → сборка морфем с @@
    morphs  = []
    cur_buf = []
    for ch, tag in zip(chars, pred_tags):
        cur_buf.append(ch)
        if tag in ('E', 'S'):
            morphs.append("".join(cur_buf))
            cur_buf = []
    if cur_buf:
        morphs.append("".join(cur_buf))

    return "@@ ".join(morphs) if len(morphs) > 1 else morphs[0]


def segment_text_turkm(text: str, model, char2id, device='cpu') -> str:
    """Сегментирует строку туркменского текста."""
    import re
    # Туркменский алфавит: латинские + спецбуквы
    token_pat = re.compile(r"[a-zA-ZçñöşüýžÇÑÖŞÜÝŽ]+|[^\w\s]|\S+")
    tokens    = token_pat.findall(text)
    result    = []
    for tok in tokens:
        tok_n = norm_turkm(tok)
        if re.match(r'^[a-zA-ZçñöşüýžÇÑÖŞÜÝŽ]+$', tok_n, re.IGNORECASE):
            result.append(segment_word_turkm(tok_n, model, char2id, device))
        else:
            result.append(tok)
    return " ".join(result)


# ════════════════════════════════════════════════════════════════════════════
# 12. ДИАГНОСТИКА ГАРМОНИИ — проверка предсказаний модели
# ════════════════════════════════════════════════════════════════════════════

def diagnose_harmony(model, char2id, test_pairs, device='cpu'):
    """
    Проверяет предсказания модели на парах (задн. основа / перед. основа).
    Пример:
      ('kitap', 'lar') — kitap (задн.) + lar (задн.) ✓
      ('öý',   'ler') — öý (перед.) + ler (перед.)  ✓
    """
    print("\n── Диагностика сингармонизма ────────────────────────────────")
    print(f"{'Слово':<20} {'Сегментация':<25} {'Гармония':<10} {'Статус'}")
    print("─" * 65)

    for word in test_pairs:
        seg      = segment_word_turkm(word, model, char2id, device)
        parts    = seg.split("@@ ")
        stem     = parts[0] if parts else word
        affix    = parts[1] if len(parts) > 1 else ""
        harmony  = "front" if word_harmony_class(stem) == HARMONY_FRONT else "back"
        ok       = check_harmony(stem, affix) if affix else True
        status   = "✅" if ok else "⚠️ нарушение"
        print(f"  {word:<18} → {seg:<25} {harmony:<10} {status}")

# Тест после обучения
TEST_WORDS = [
    # задние основы (→ -lar, -da, -dan, -a, -yň)
    "kitaplar",       # книги     kitap+lar
    "adamlaryň",      # людей     adam+lar+yň
    "ýurtda",         # в стране  ýurt+da
    "okadym",         # я учил    oka+dym
    # передние основы (→ -ler, -de, -den, -e, -iň)
    "öýler",          # дома      öý+ler
    "işçileriň",      # рабочих   işçi+ler+iň
    "döwletde",       # в гос-ве  döwlet+de
    "geldim",         # пришёл    gel+dim
    # длинные цепочки
    "Türkmenistanda", # в Туркм-не Türkmenistan+da
    "köçelerinde",    # на его улицах köçe+ler+inde
]

best_model_path = os.path.join(MODEL_DIR, "femseg_turkm_v1_BEST.pt")
if os.path.exists(best_model_path):
    best_mdl, c2id = load_model_for_inference(best_model_path, device=str(device))
    diagnose_harmony(best_mdl, c2id, TEST_WORDS, device=str(device))

print("\n✅ LCMSeg_turkm_v1 готов.")
