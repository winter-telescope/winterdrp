import astropy.io.fits
import numpy as np
from abc import ABC

import pandas as pd

from winterdrp.processors.base_processor import BaseImageProcessor, BaseDataframeProcessor
import logging
from winterdrp.processors.database.postgres import DataBaseError, export_to_db
from winterdrp.processors.database.base_database_processor import BaseDatabaseProcessor

logger = logging.getLogger(__name__)


class BaseDatabaseExporter(BaseDatabaseProcessor, ABC):
    base_key = "dbexporter"


class DatabaseImageExporter(BaseDatabaseExporter, BaseImageProcessor):

    def _apply_to_images(
            self,
            images: list[np.ndarray],
            headers: list[astropy.io.fits.Header],
    ) -> tuple[list[np.ndarray], list[astropy.io.fits.Header]]:

        for header in headers:
            primary_keys, primary_key_values = export_to_db(
                header,
                db_name=self.db_name,
                db_table=self.db_table,
                db_user=self.db_user,
                password=self.db_password
            )

            for ind, key in enumerate(primary_keys):
                header[key] = primary_key_values[ind]
        return images, headers


class DatabaseDataframeExporter(BaseDatabaseExporter, BaseDataframeProcessor):

    def _apply_to_candidates(
            self,
            candidate_table: pd.DataFrame
    ) -> pd.DataFrame:

        new_table = pd.DataFrame()

        for index, candidate_row in candidate_table.iterrows():
            primary_keys, primary_key_values = export_to_db(
                candidate_row.to_dict(),
                db_name=self.db_name,
                db_table=self.db_table,
                db_user=self.db_user,
                password=self.db_password
            )
            for ind, key in enumerate(primary_keys):
                candidate_row[key] = primary_key_values[ind]
            new_table.append(candidate_row, ignore_index=True)

        return new_table

