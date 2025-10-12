package com.xiaozhi.vo;

import java.util.List;

/**
 * 分页结果VO
 */
public class PageResult<T> {
    private List<T> list;
    private Pagination pagination;

    public PageResult() {}

    public PageResult(List<T> list, Pagination pagination) {
        this.list = list;
        this.pagination = pagination;
    }

    public List<T> getList() {
        return list;
    }

    public void setList(List<T> list) {
        this.list = list;
    }

    public Pagination getPagination() {
        return pagination;
    }

    public void setPagination(Pagination pagination) {
        this.pagination = pagination;
    }

    public static class Pagination {
        private Integer page;
        private Integer perPage;
        private Long total;
        private Integer pages;

        public Pagination() {}

        public Pagination(Integer page, Integer perPage, Long total) {
            this.page = page;
            this.perPage = perPage;
            this.total = total;
            this.pages = (int) Math.ceil((double) total / perPage);
        }

        public Integer getPage() {
            return page;
        }

        public void setPage(Integer page) {
            this.page = page;
        }

        public Integer getPerPage() {
            return perPage;
        }

        public void setPerPage(Integer perPage) {
            this.perPage = perPage;
        }

        public Long getTotal() {
            return total;
        }

        public void setTotal(Long total) {
            this.total = total;
        }

        public Integer getPages() {
            return pages;
        }

        public void setPages(Integer pages) {
            this.pages = pages;
        }
    }
}