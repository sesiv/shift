# Changelog

Формат основан на [Keep a Changelog](https://keepachangelog.com/ru/1.1.0/) и следует принципам [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-11-07
### Added
- utils.py - файл с вспомогательными функциями
- consts.py - файл с константами

### Changed
- вынесены константы в src/app/consts.py
- вынесены функции из main.py в utils.py

## [0.2.0] - 2025-11-02

### Changed
- e5.py и vector_db.py теперь общаются не через http а через прямой импорт

### Removed
- Удалены зависимости e5.py и его docker build , эти функции перешли на vector_db.py


## [0.1.0] - 2025-10-22

### Added
- Создан новый репозиторий Service_Desk