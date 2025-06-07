import React from 'react';

const Pagination = ({ currentPage, totalPages, onChangePage }) => {
    const pages = Array.from({ length: totalPages }, (_, i) => i + 1);

    return (
        <div className="pagination">
            {pages.map(num => (
                <button
                    key={num}
                    className={currentPage === num ? 'active' : ''}
                    onClick={() => onChangePage(num)}
                >
                    {num}
                </button>
            ))}
        </div>
    );
};

export default Pagination;
