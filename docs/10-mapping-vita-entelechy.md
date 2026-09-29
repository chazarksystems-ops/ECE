# Mapping onto Vita, Firebreak, Entelechy

These are product boundaries, not a merge plan.

## Vita / Firebreak

Vita is a Conway-line sandbox (discrete Life + canopy / heightmap /
JSON export / Firebreak mechanics). That is Life-like + optional
continuous height, not Mohr.

| Vita need | ECE piece |
|---|---|
| Mass that does not evaporate | MaCE / Flow-Lenia transport |
| Discrete Life | separate grid update, not the tent |
| Firebreak as a barrier field | extra Eulerian channel sampled by the CA |
| JSON exports | dump `ρ` and rule params, not particle SoA |

Do not drive Vita’s tick with Mohr forces. If a hybrid is wanted,
Firebreak is a field that the CA reads, not a species in `A`.

## Entelechy

Entelechy is the visual graph workbench (React Flow ports / handles /
wires, JSON lane packets, object registry). ECE rules become graph
nodes later; they are not nodes today.

Suggested port types when the graph wraps ECE:

| Port | Buffer |
|---|---|
| `particles` | SoA pos/vel/type |
| `field_c` | one Eulerian channel |
| `potential` | convolution output |
| `matrix` | K×K |
| `kernel` | radial samples |
| `ranges` | hash table |

Edges: `deposit`, `sample`, `potential`, `force`. A lane packet can
carry a `SimConfig` TOML blob plus a matrix checksum.

Daily Entelechy JSON packs stay in that repo. This repo stays the
sim kernel + spec.

## PlasmaForge / PIC

Deposit / sample is the same scatter–gather shape as charge deposit
in a PIC plasma step. Share the pass idea, not the binary. Plasma
needs Poisson / Maxwell; ALife does not.

## SUBTERRA / game work

Game ticks can call M1 Mohr as a toy ecosystem layer. They must not
take a dependency on MaCE 3D or the evaluator.
