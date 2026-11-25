// QZ_NC: the tensor-parallel card count the arch code is compiled for (2 by default).
// Arch code reads it at compile time; shared headers never do, so the same shared code
// serves arch builds for different counts.
#pragma once

#ifndef QZ_NC
#define QZ_NC 2
#endif
static_assert(QZ_NC == 2 || QZ_NC == 4, "QZ_NC: 2 or 4 cards");
