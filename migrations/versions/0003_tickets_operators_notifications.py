"""Operators, tickets and the notification outbox.

Schema mirrors ``db/models.py`` exactly (verified in tests with Alembic's
``compare_metadata``). Three demo operators are seeded so assignment works out
of the box; replace them with your real support staff.

Revision ID: 0003
Revises: 0002
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "operators",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("email", sa.String(254), nullable=False, unique=True),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
    )
    op.create_table(
        "tickets",
        sa.Column("id", sa.String(12), primary_key=True),
        sa.Column("title", sa.String(120), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("priority", sa.String(10), nullable=False),
        sa.Column("status", sa.String(12), nullable=False, server_default="open"),
        sa.Column("customer_id", sa.String(16)),
        sa.Column("assignee_id", sa.Integer(), sa.ForeignKey("operators.id", ondelete="SET NULL")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("closed_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint("status IN ('open', 'in_progress', 'closed')", name="ck_tickets_status"),
        sa.CheckConstraint(
            "priority IN ('low', 'medium', 'high', 'urgent')", name="ck_tickets_priority"
        ),
    )
    op.create_index("ix_tickets_status", "tickets", ["status"])
    op.create_index("ix_tickets_created_at", "tickets", ["created_at"])
    op.create_index("ix_tickets_closed_at", "tickets", ["closed_at"])

    op.create_table(
        "notifications",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "ticket_id",
            sa.String(12),
            sa.ForeignKey("tickets.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("channel", sa.String(10), nullable=False),
        sa.Column("status", sa.String(10), nullable=False, server_default="pending"),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("last_error", sa.Text()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("sent_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint("channel IN ('email', 'slack')", name="ck_notifications_channel"),
        sa.CheckConstraint(
            "status IN ('pending', 'sent', 'failed')", name="ck_notifications_status"
        ),
    )
    op.create_index("ix_notifications_status_created", "notifications", ["status", "created_at"])

    op.execute("""
        INSERT INTO operators (name, email) VALUES
            ('Maya Cohen', 'maya.cohen@support.example'),
            ('Liam Novak', 'liam.novak@support.example'),
            ('Aisha Rahman', 'aisha.rahman@support.example')
        ON CONFLICT (email) DO NOTHING
        """)


def downgrade() -> None:
    op.drop_table("notifications")
    op.drop_table("tickets")
    op.drop_table("operators")
